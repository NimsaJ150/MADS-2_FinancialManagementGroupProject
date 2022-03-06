import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import openpyxl
from sklearn.linear_model import LinearRegression

"""
Load data
"""

# import data
ceo_data_raw = pd.read_csv("data/Execucomp_2006-2021.csv")  # "data/Data_by_CEO.xlsx", )
company_data_raw = pd.read_csv("data/CCM_Fundamentals_Annual_2006_-_2021_new.csv")
price_data_raw = pd.read_csv("data/return_annual.csv")

company_data_raw.head()
ceo_data_raw.head()

company_data_raw_columns = company_data_raw.columns
company_cols = ['GVKEY', 'prcc_f', 'ajex', 'ajp', 'fyear']
ceo_data_raw_columns = ceo_data_raw.columns
ceo_cols = ['GVKEY', 'CO_PER_ROL', 'YEAR', 'AGE', 'BECAMECEO', 'TITLE', 'PCEO', 'LEFTOFC']
price_data_raw_columns = price_data_raw.columns
price_cols = []

# filter data
company_data = company_data_raw[company_cols]

ceo_data = ceo_data_raw[ceo_cols]
ceo_data = ceo_data[ceo_data.PCEO == "CEO"]

price_data = price_data_raw[ceo_cols]

# join data
data_joined = ceo_data.join(company_data.set_index(['GVKEY', 'fyear']), on=['GVKEY', 'YEAR'], how='left', lsuffix='',
                            rsuffix='', sort=False)

data_joined = data_joined.join(price_data.set_index(['LPERMNO', 'year']), on=['LPERMNO', 'YEAR'], how='left', lsuffix='',
                               rsuffix='', sort=False)

"""
Preprocess data
"""

# drop nan in prcc_f or ajex/ajp columns
data_joined.dropna(inplace=True)

data_joined.apply(lambda x: x.astype(str).str.lower())

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

# drop columns only important for joining
data_joined.drop(['GVKEY', 'CO_PER_ROL', 'PCEO', 'ajex', 'ajp', 'TITLE'], axis=1, inplace=True)

"""
Additional features
"""
# 3 year requirement for managers
# BECAMECEO LEFTOFC

# how many years as CEO - ceo_tenure
# how many years working there -
# percentage change in stock prices

# -> group for single CEOs - avg_change_in_sp

"""
Fixed effects
"""



"""
Linear Regression
"""

X = data_joined.drop('prcc_f', axis=1)
y = data_joined['prcc_f']
lr = LinearRegression()

lr.fit(X, y)
lr.coef_

coefficients = pd.concat([pd.DataFrame(X.columns), pd.DataFrame(np.transpose(lr.coef_))], axis=1)
coefficients
