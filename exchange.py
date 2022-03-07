import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import openpyxl
from sklearn.linear_model import LinearRegression

"""
Load data
"""

# import data
ceo_data_raw = pd.read_csv("data/Execucomp_2006_-_2021_MTL.csv")
company_data_raw = pd.read_csv("data/CCM_Fundamentals_Annual_2006_-_2021_new.csv")
price_data_raw = pd.read_csv("data/return_annual.csv")

company_data_raw.head()
ceo_data_raw.head()

company_data_raw_columns = company_data_raw.columns
company_cols = ['GVKEY', 'LPERMNO', 'prcc_f', 'fyear']
ceo_data_raw_columns = ceo_data_raw.columns
ceo_cols = ['GVKEY', 'CO_PER_ROL', 'YEAR', 'AGE', 'BECAMECEO', 'TITLE', 'CEOANN', 'LEFTOFC']
price_data_raw_columns = price_data_raw.columns
price_cols = []

# filter data
company_data = company_data_raw[company_cols]
company_data.loc['fyear'] = company_data['fyear'].astype(int) #--gives a warning, value set to copy instead of view

ceo_data = ceo_data_raw[ceo_cols]
ceo_data = ceo_data[ceo_data.CEOANN == "CEO"]

#drop age with nans - about 100 rows removed
ceo_data = ceo_data[ceo_data['AGE'].notna()]

price_data = price_data_raw

# join data

data_joined = company_data.join(price_data.set_index(['LPERMNO', 'year']), on=['LPERMNO', 'fyear'], how='inner',
                                lsuffix='',
                                rsuffix='', sort=False)


data_joined = data_joined.join(ceo_data.set_index(['GVKEY', 'YEAR']), on=['GVKEY', 'fyear'], how='inner', lsuffix='',
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
data_joined.drop(['GVKEY', 'CO_PER_ROL', 'TITLE'], axis=1, inplace=True) #ajex, ajp removed

"""
Additional features
"""
# 3 year requirement for managers
# BECAMECEO LEFTOFC
data_joined['BECAMECEO'] = pd.to_datetime(data_joined['BECAMECEO'], format='%Y%m%d')
data_joined['LEFTOFC'] = pd.to_datetime(data_joined['LEFTOFC'], format='%Y%m%d')

# data_joined['3Y_THRESH'] = (data_joined['LEFTOFC']-data_joined['BECAMECEO'])/365.35 #incomplete
# data_joined['3Y_THRESH'] >= 3


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
# X = data_joined.drop('prcc_f', axis=1)
# y = data_joined['prcc_f']
# lr = LinearRegression()
#
# lr.fit(X, y)
# lr.coef_
#
# coefficients = pd.concat([pd.DataFrame(X.columns), pd.DataFrame(np.transpose(lr.coef_))], axis=1)
# coefficients
