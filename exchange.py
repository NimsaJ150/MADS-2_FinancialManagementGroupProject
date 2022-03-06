#%%

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import openpyxl
from sklearn.linear_model import LinearRegression

#%% md

# Load data

#%%

# import data
company_data_raw = pd.read_csv("data/GSFM_CRSP_Compustat_2006_-_2021.csv")
ceo_data_raw = pd.read_csv("data/Execucomp_2006-2021.csv")  # "data/Data_by_CEO.xlsx", )

#%%
ceo_data_raw.head()

#%%

company_data_raw.head()

#%%

company_data_raw_columns = company_data_raw.columns
company_cols = ['GVKEY', 'prcc_f', 'ajex', 'ajp', 'fyear']

#%%

ceo_data_raw_columns = ceo_data_raw.columns
ceo_cols = ['GVKEY', 'CO_PER_ROL', 'YEAR', 'AGE', 'BECAMECEO', 'TITLE', 'PCEO', 'LEFTOFC']

#%%

# filter data
company_data = company_data_raw[company_cols]

ceo_data = ceo_data_raw[ceo_cols]
a = ceo_data[ceo_data.PCEO == "CEO"]

#%%

# join data
data_joined = ceo_data.join(company_data.set_index(['GVKEY', 'fyear']), on=['GVKEY', 'YEAR'], how='left', lsuffix='',
                            rsuffix='', sort=False)

#%%
data_joined.head()
#%% md

# Preprocess data

#%%

# drop nan in prcc_f or ajex/ajp columns
data_joined.dropna(inplace=True)

data_joined.apply(lambda x: x.astype(str).str.lower())

#%%

# Chairman, President, Founder

# founder - 1, otherwise 0
if_founder = data_joined['TITLE'].str.contains('founder')
data_joined['dummy_founder'] = if_founder

# president - 1, otherwise 0
if_president = data_joined['TITLE'].str.contains('president')
data_joined['dummy_president'] = if_founder

# chairman - 1, otherwise 0
if_chairman = data_joined['TITLE'].str.contains('chairman')
data_joined['dummy_chairman'] = if_founder

# chairman + president - 1, otherwise 0
data_joined['dummy_chairman_president'] = data_joined['TITLE'].str.contains('|'.join(['chairmam', 'president']))

#%%

# drop columns only important for joining
data_joined.drop(['GVKEY', 'CO_PER_ROL', 'PCEO', 'ajex', 'ajp', 'TITLE'], axis=1, inplace=True)

#%%
# 3 year requirement for managers
# BECAMECEO LEFTOFC

#%% md

# Additional Features

#%%

# how many years as CEO - ceo_tenure
# how many years working there -
# percentage change in stock prices

# -> group for single CEOs - avg_change_in_sp

#%% md

# Linear regression

#%%

X = data_joined.drop('prcc_f', axis=1)
y = data_joined['prcc_f']
lr = LinearRegression()

#%%

lr.fit(X, y)
lr.coef_

#%%

coefficients = pd.concat([pd.DataFrame(X.columns), pd.DataFrame(np.transpose(lr.coef_))], axis=1)
coefficients

#%%


