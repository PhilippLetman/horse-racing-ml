#!/usr/bin/env python
# coding: utf-8

# In[4]:


'''
This notebook is for cleaning the data and preparing the data for feature engineering

'''
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from pathlib import Path
    import time
    import subprocess
    from datetime import date
    import pyarrow as pa, pyarrow.parquet as pq
    import sys
    
    import xgboost as xgb
    import lightgbm as lgb
    import optuna
    print('The libraries have been successfully installed')
except ImportError as e:
    print(f"❌ Missing library: {e.name}")


# In[ ]:


DATA_ROOT = Path.home() / 'scrape_data' / 'data'
GB_DIR = DATA_ROOT / 'gb' / 'flat'
IRE_DIR = DATA_ROOT / 'ire' / 'flat'
folders = [GB_DIR, IRE_DIR]

def stream_data(folders, out_path, chunk_size = 150_000):
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for folder in folders:
        for f in sorted(Path(folder).glob("*.csv")):
            for chunk in pd.read_csv(f, chunksize=chunk_size):
                chunk['region']=Path(folder).parent.name
                chunk['year']=int(f.stem)
                for c in ('num','draw'):
                    if c in chunk:
                        chunk[c]=pd.to_numeric(chunk[c], errors = 'coerce').astype('Int16')
                if 'prize' in chunk:
                    chunk['prize'] = (chunk['prize'].astype('string').str.replace(r"[^\d]","", regex=True).replace("",pd.NA).pipe(pd.to_numeric, errors='coerce').astype('float32'))
                if 'class' in chunk:
                    chunk['class']= chunk['class'].astype('string')
                chunk = fix_prob_col(chunk)
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
    if writer is not None:
        writer.close()
stream_data(folders,DATA_ROOT / 'all_flat.parquet')


# In[12]:


DATA_ROOT = Path.home() / 'scrape_data' / 'data'
df_check = pd.read_parquet(DATA_ROOT / "all_flat.parquet")
print(df_check.columns.tolist()[:12])
print(df_check.head(5).iloc[:, :8])
len(df_check)


# In[28]:


'''
Now below I will clean and sort the data which I have obtained
'''
df=df_check.copy()

def clean(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()]
    df['dist_f']= df['dist_f'].astype('string').str.replace('f', '', regex = False)
    df['year'] = df['date'].dt.year
    int_cols = ['dist_m','ran','num','draw','age','lbs','or','rpr']
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors = 'coerce').astype('Int16')
    num_cols = ['secs','btn','ovr_btn','dec','dist_f']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors = 'coerce').astype('float32')

    df['pos_str'] = df['pos'].astype('string')
    df['pos_num'] = pd.to_numeric(df['pos'], errors = 'coerce').astype('Int16')

    cat_cols = ['region','pattern','course','sex_rest','race_name','going','surface','horse','hg','jockey','trainer','dam','damsire','sire','owner']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    remove_cols = ['type', 'class', 'dist']
    df = df.drop(columns=remove_cols)
    return df

def clean_prize(df):
    if 'prize' not in df:
        return df
    df = df.copy()
    df['currency'] = df['region'].str.upper().map({'IRE':"EUR",'GB':"GBP"}).astype("category")
    s = (df['prize'].astype('string').str.strip())
    mask_ire = df['region'].eq('IRE')
    s_ire = s.where(mask_ire, None)
    s_ire = s_ire.str.replace("€", "", regex=False)
    s_clean = s.where(~mask_ire, s_ire)
    major_units = pd.to_numeric(s_clean, errors = 'coerce').astype('float32')
    df["prize_money"]= major_units 
    df['prize_arith'] = (major_units * 100).round().astype('Int64')
    return df

def off_time_features(df):
    s = df['off'].astype('string').str.strip()
    df['post_time'] = pd.to_datetime(df['date'].dt.strftime('%Y-%m-%d') + " " + s, errors='coerce')
    df['off_min'] = (df['post_time'].dt.hour *60 + df['post_time'].dt.minute).astype('Int16')
    return df

def parse_age_band(df):
    s = df['age_band'].astype('string').str.strip().str.lower()
    s = (s.str.replace(r'\s+'," ", regex = True).str.replace('yo', " ", regex=False))

    band_min = pd.Series(pd.NA, index=df.index, dtype='Int16')
    band_max = pd.Series(pd.NA, index=df.index, dtype='Int16')
    band_plus = pd.Series(0, index=df.index, dtype="Int8")

    m = s.str.extract(r"^\s*(?P<min>\d+)\s*\+\s*$")
    mask = m["min"].notna()
    band_min.loc[mask] = m.loc[mask, "min"].astype("Int16")
    band_plus.loc[mask] = 1

    m = s.str.extract(r"^\s*(?P<min>\d+)\s*[-–]\s*(?P<max>\d+)\s*$")
    mask = m["min"].notna()
    band_min.loc[mask] = m.loc[mask, "min"].astype("Int16")
    band_max.loc[mask] = m.loc[mask, "max"].astype("Int16")

    m = s.str.extract(r"^\s*(?P<eq>\d+)\s*$")
    mask = m["eq"].notna() & band_min.isna()
    band_min.loc[mask] = m.loc[mask, "eq"].astype("Int16")
    band_max.loc[mask] = m.loc[mask, "eq"].astype("Int16")

    df["band_min_age"] = band_min
    df["band_max_age"] = band_max
    df["is_band_plus"] = band_plus

    if "age" in df.columns:
        age_num = pd.to_numeric(df["age"], errors="coerce").astype("Int16")
        df["age_minus_min"] = (age_num.astype("float32") - band_min.astype("float32")).astype("float32")
        df["age_to_max"] = np.where(band_max.notna(),
                                    (band_max.astype("float32") - age_num.astype("float32")),
                                    np.nan).astype("float32")
    else:
        df["age_minus_min"] = np.nan
        df["age_to_max"] = np.nan

    return df 


# In[29]:


pre_rows = len(df_check)

df_clean = (df_check.copy()
            .pipe(clean)
            .pipe(clean_prize)
            .pipe(off_time_features)
            .pipe(parse_age_band))

post_rows = len(df_clean)


# In[30]:


df_clean.to_parquet(DATA_ROOT / "all_flat_clean.parquet")


# In[31]:


print(df_clean.iloc[5,:])
print(df_clean[['prize_money']].head(5))


# In[ ]:




