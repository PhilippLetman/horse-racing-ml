#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
This project is for developing a machine learning model to predict horse race outcomes.  

Goals:
- Store race data in relational format (Races, Horses, Jockeys, Trainers, Results).
- Perform feature engineering (form, strike rates, etc.).
- Train baseline and advanced models (logistic regression, XGBoost, LightGBM).
- Evaluate with accuracy, log loss, and betting strategy simulations.

Data was collected using a modified version of the rpscrape tool by joenano. 
I adapted it for my workflow, but the rpscrape source code is not included in this repository. 
It is run locally to produce CSV outputs that are then cleaned and analysed here.
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


# In[3]:


'''
Bulk scraping of the data from all the years 2007-2025 and also pre-2007
'''


# In[4]:


REGION = "gb"
SURFACE = "flat"
YEARS = range(2007,2026)
REPO = Path(os.environ.get("SCRAPE_DIR", str(Path.home() / "scrape_data")))
WORKDIR = REPO / "scripts"
SCRIPT = WORKDIR / "rpscrape1.py"
OUTDIR = Path("external_data") / "dates" / REGION
OUTDIR.mkdir(parents=True, exist_ok=True)
PY = sys.executable

def scrape_year(y: int):
    start = date(y, 1, 1)
    end = date(y, 12, 31)
    daterange = f"{start:%Y/%m/%d}-{end:%Y/%m/%d}"

    outfile = OUTDIR / f"{REGION}_{SURFACE}_{y}.csv"
    logfile = OUTDIR / f"{REGION}_{SURFACE}_{y}.log"
    if outfile.exists():
        print(f"Skip {outfile.name}")
        return

    cmd = [PY, SCRIPT.name, "-r", REGION, "-y", str(y), "-t", SURFACE]
    print("Running:"," ".join(cmd), "in", WORKDIR)
    with outfile.open('w', encoding ='utf-8') as f_out, logfile.open('w', encoding = 'utf-8') as f_log:
        try:
            subprocess.run(cmd, cwd=str(WORKDIR), stdout=f_out, stderr=f_log, text=True, check=True)
            print(f"Saved {outfile.name}")
        except subprocess.CalledProcessError:
            print(f"Failed {y}. See {logfile.name}")
for year in YEARS:
    scrape_year(year)
    time.sleep(2)        


# In[ ]:




