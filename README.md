# horse-racing-ml
In the process of building a machine learning pipeline to predict outcomes of flat horse races in UK and Ireland using historical data 

01_data_prep:
The race data used in this project was collected using a modified version of the rpscrape tool which is originally developed by joenano.
I adapted parts of the scraper to suit my own requirements (e.g. adjusting file paths, adding retry behaviour and data-saving logic) but the core structure remains based on the original tool. The rpscrape source code is not included in this repository but it is executed locally to produce the CSV outputs.

02_cleaning_data:
This notebook first consolidates all CSVs into one parquet file. Next, it cleans and normalises the data, while also adding extra feautures that will be useful for predictions. 


