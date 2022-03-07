import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

"""
Load data
"""

# import data
ceo_data_raw = pd.read_csv("data/Execucomp_2006_-_2021_MTL.csv")
company_data_raw = pd.read_csv("data/CCM_Fundamentals_Annual_2006_-_2021_new.csv")
price_data_raw = pd.read_csv("data/return_annual.csv")

company_data_raw_columns = company_data_raw.columns
company_cols = ['GVKEY', 'LPERMNO', 'prcc_f', 'fyear']
ceo_data_raw_columns = ceo_data_raw.columns
ceo_cols = ['GVKEY', 'CO_PER_ROL', 'YEAR', 'AGE', 'TITLE', 'PCEO', 'BECAMECEO', 'LEFTOFC', 'LEFTCO', 'JOINED_CO']
price_data_raw_columns = price_data_raw.columns
price_cols = []

# filter data
company_data = company_data_raw[company_cols]
company_data['fyear'] = company_data['fyear'].astype(int)   #  SettingWithCopyWarning: 
# A value is trying to be set on a copy of a slice from a DataFrame. 
# Try using .loc[row_indexer,col_indexer] = value instead


ceo_data = ceo_data_raw[ceo_cols]
ceo_data = ceo_data[ceo_data.PCEO == "CEO"]

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

# drop nan in prcc_f or ajex/ajp columns --> this is deleting everything except 6 rows??
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
data_joined.drop(['GVKEY', 'CO_PER_ROL', 'PCEO', 'TITLE'], axis=1, inplace=True) #ajex, ajp removed


"""
Additional features
"""
# 3 year requirement for managers
# BECAMECEO LEFTOFC
# data_joined['BECAMECEO'] = pd.to_datetime(data_joined['BECAMECEO'], format='%Y%m%d')
# data_joined['LEFTOFC'] = pd.to_datetime(data_joined['LEFTOFC'], format='%Y%m%d')

# data_joined['3Y_THRESH'] = (data_joined['LEFTOFC']-data_joined['BECAMECEO'])/365.35 #incomplete
# data_joined['3Y_THRESH'] >= 3


# print(data_joined["JOINED_CO"].isnull().sum())
# print(data_joined["JOINED_CO"])

# how many years as CEO - ceo_tenure
# PROBLEM - NULL VALUES, negative values?? :)
temp = pd.DataFrame(data_joined, columns=['BECAMECEO', 'LEFTOFC', 'LEFTCO', 'JOINED_CO'])
temp['LEFTOFC'] = pd.to_datetime(temp['LEFTOFC'], format='%Y%m%d')
temp['BECAMECEO'] = pd.to_datetime(temp['BECAMECEO'], format='%Y%m%d')
temp['LEFTCO'] = pd.to_datetime(temp['LEFTCO'], format='%Y%m%d')
temp['JOINED_CO'] = pd.to_datetime(temp['JOINED_CO'], format='%Y%m%d')

data_joined['ceo_tenure'] = (temp['LEFTOFC'] - temp['BECAMECEO']).dt.days
# print(data_joined['ceo_tenure'])

# how many years working there 
# PROBLEM - NULL VALUES
data_joined['working_days'] = (temp['LEFTCO'] - temp['JOINED_CO']).dt.days
# print(data_joined['working_days'])

# percentage change in stock prices

# -> group for single CEOs - avg_change_in_sp

# drop columns only important for calculating date related attributes
data_joined.drop(['JOINED_CO', 'BECAMECEO', 'LEFTCO', 'LEFTOFC'], axis=1, inplace=True)


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
