import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

"""
Load data
"""
# import data
ceo_data_raw = pd.read_csv("data/Execucomp_2006_-_2021_MTL.csv")
company_data_raw = pd.read_csv("data/CCM_Fundamentals_Annual_2006_-_2021_NEW_winsor.csv")
annual_return_price_data_raw = pd.read_csv("data/Compustat_annual_return.csv")
annual_stdev_price_data_raw = pd.read_csv("data/CRSP_Standard_Deviation_by_Year.csv")

company_data_raw.head()
ceo_data_raw.head()

company_data_raw_columns = company_data_raw.columns
# company_cols = ['GVKEY', 'LPERMNO', 'prcc_f', 'fyear', 'ROA', 'Tobins_Q', 'Cash_Flow',
#                 'Leverage', 'Investment', 'Cash_Holdings', 'Div_over_Earn', 'SG_A', 'aqc']
company_cols = ['GVKEY', 'LPERMNO', 'prcc_f', 'fyear', 'ROA_w01_w99', 'Tobins_Q_w95', 'CF_w01_w99',
                'Lev_w05_w95', 'Inv_w01_w99', 'CashHold_w95', 'DoE_w01_w99', 'Acquisitions']
ceo_data_raw_columns = ceo_data_raw.columns
ceo_cols = ['GVKEY', 'CO_PER_ROL', 'YEAR', 'AGE', 'BECAMECEO', 'TITLE', 'CEOANN', 'LEFTOFC', 'LEFTCO', 'JOINED_CO',
            'CONAME', 'EXECID']

annual_return_cols = ['LPERMNO', 'fyear', 'prcc_pcchg']
annual_stdev_cols = ['LPERMNO', 'year', 'stand_dev']

# filter data
company_data = company_data_raw[company_cols]
# company_data['fyear'] = company_data['fyear'].astype(int)  # --gives a warning, value set to copy instead of view
#  SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

ceo_data = ceo_data_raw[ceo_cols]
ceo_data = ceo_data[ceo_data.CEOANN == "CEO"]
# --------pls dont delete this (Irene)
# drop age with nans - about 100 rows removed
ceo_data = ceo_data[ceo_data['AGE'].notna()]

annual_return_price_data = annual_return_price_data_raw[annual_return_cols]
annual_return_price_data.rename(columns={'prcc_pcchg': 'avg_return'}, inplace=True)

annual_stdev_price_data = annual_stdev_price_data_raw[annual_stdev_cols]
annual_stdev_price_data.rename(columns={'stand_dev': 'sd_return'}, inplace=True)


# join data
data_joined = company_data.join(annual_return_price_data.set_index(['LPERMNO', 'fyear']), on=['LPERMNO', 'fyear'], how='inner',
                                lsuffix='',
                                rsuffix='', sort=False)

data_joined = data_joined.join(annual_stdev_price_data.set_index(['LPERMNO', 'year']), on=['LPERMNO', 'fyear'], how='inner',
                                lsuffix='',
                                rsuffix='', sort=False)

data_joined = data_joined.join(ceo_data.set_index(['GVKEY', 'YEAR']), on=['GVKEY', 'fyear'], how='inner', lsuffix='',
                               rsuffix='', sort=False)

# drop rows with 2021 - because there are some values for PRCC_F data missing
# print(data_joined.shape)
data_joined = data_joined[data_joined["fyear"] != 2021]
# print(data_joined.shape)

"""
Preprocess data
"""

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
data_joined.drop(['TITLE'], axis=1, inplace=True)

# expect CEOs with nan in the title column to be only CEO and having no other position
data_joined['dummy_founder'].fillna(False)
data_joined['dummy_president'].fillna(False)
data_joined['dummy_chairman'].fillna(False)
data_joined['dummy_chairman_president'].fillna(False)

"""
Additional features
"""


# print(data_joined.shape)
# print(data_joined.isnull().sum(axis=0))  # return number of nan vals for each column

# fixing the date stuff
# if ceo has Nan for BECAMECEO, we check if it exists when grouping by CO_PER_ROL or set it to JOIN_CO if it exists
def fix_becameceo(row):
    if pd.isnull(row['BECAMECEO']):
        return row['JOINED_CO']
    return row['BECAMECEO']


# print(data_joined['BECAMECEO'].isnull().sum())

# checking if the date exists by grouping --> it doesn't
# ceo_group = pd.DataFrame()
# ceo_group['BECAMECEO'] = data_joined.groupby('CO_PER_ROL')['BECAMECEO'].apply(lambda ser: ser.unique())
# print(ceo_group['BECAMECEO'].isnull().sum())

# setting the start date as the JOINED_CO date --> from 282 Nan values to 265 (should we drop others??)
data_joined['BECAMECEO'] = data_joined.apply(lambda row: fix_becameceo(row), axis=1)


# print(data_joined['BECAMECEO'].isnull().sum())

# if ceo has Nan for LEFTOFC, we set it to 31.12.2020 - we assume the person is still the CEO of the company
# we excluded 2021 data, that's why we used 2020
def fix_leftofc(row):
    if pd.isnull(row['LEFTOFC']):
        return '20201231'
    return row['LEFTOFC']


# print(data_joined['LEFTOFC'].isnull().sum()) # 8975 Nan values
# ceo_group = pd.DataFrame()
# ceo_group['LEFTOFC'] = data_joined.groupby('CO_PER_ROL')['LEFTOFC'].apply(lambda ser: ser.unique())  # doesn't help
# print(ceo_group['LEFTOFC'].isnull().sum())

data_joined['LEFTOFC'] = data_joined.apply(lambda row: fix_leftofc(row), axis=1)
# print(data_joined['LEFTOFC'].isnull().sum()) # 0 Nan

# 3 year requirement for managers-----------------------------
data_joined['BECAMECEO'] = pd.to_datetime(data_joined['BECAMECEO'], format='%Y%m%d')
data_joined['LEFTOFC'] = pd.to_datetime(data_joined['LEFTOFC'], format='%Y%m%d')

# too much nans - we exclude this two columns in the end of additional features
# data_joined['JOINED_CO'] = pd.to_datetime(data_joined['BECAMECEO'], format='%Y%m%d')
# data_joined['LEFTCO'] = pd.to_datetime(data_joined['LEFTOFC'], format='%Y%m%d')

data_joined['3Y_THRESH'] = data_joined['LEFTOFC'].dt.year - data_joined['BECAMECEO'].dt.year
data_joined = data_joined[data_joined['3Y_THRESH'] >= 3]  # lol no change because all ceo's so far have stayed>= 3 yr

# drop column used for 3yr requirement
data_joined.drop(['3Y_THRESH'], axis=1, inplace=True)
# --------------------------------------------------------------
# at least 2 company requiremnt for managers -------------------
# for each EXEC ID - at least 2 distinct GVKEY
tempdf = data_joined
var = data_joined[data_joined['EXECID'] == 17580]  # manually checking

ceo_group = data_joined.groupby('EXECID')['GVKEY'].nunique()
ceo_group.name = 'ceo_group'

data_joined = data_joined.join(ceo_group, on=['EXECID'], how='inner', lsuffix='',
                               rsuffix='', sort=False)

# --------------------------------------------------------------


# how many years as CEO - ceo_tenure ----------------------------------
data_joined['ceo_tenure'] = (data_joined['LEFTOFC'] - data_joined['BECAMECEO']).dt.days

# how many years working there - not relevant anymore -----------------
# PROBLEM - NULL VALUES
# data_joined['working_days'] = (data_joined['LEFTCO'] - data_joined['JOINED_CO']).dt.days
# print(data_joined['working_days'])

# percentage change in stock prices

# -> group for single CEOs - avg_change_in_sp

# drop columns only important for calculating date related attributes
data_joined.drop(
    ['JOINED_CO', 'BECAMECEO', 'LEFTCO', 'LEFTOFC', 'CO_PER_ROL', 'CEOANN', 'CONAME', 'LPERMNO', 'prcc_f', 'EXECID'],
    axis=1,
    inplace=True)

# removing nan rows in the end
# print(data_joined.shape)
# print(data_joined.isnull().sum(axis=0))
data_joined.dropna(inplace=True)
# print(data_joined.shape)

"""
Fixed effects models
"""

from linearmodels import PanelOLS
import statsmodels.api as sm

###########################
# models for predicting avg_return (all vars)

data_model = data_joined.set_index(['GVKEY', 'fyear', ]).copy(deep=True)

data_model.drop('sd_return', axis=1, inplace=True)
X = sm.add_constant(data_model.drop('avg_return', axis=1))
y = data_model.avg_return

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=True, time_effects=True)
re_res = mod.fit()
print(re_res)

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=True, time_effects=False)
re_res = mod.fit()
print(re_res)

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=False, time_effects=True)
re_res = mod.fit()
print(re_res)

###########################
# models for predicting sd_return (all vars)

data_model = data_joined.set_index(['GVKEY', 'fyear']).copy(deep=True)

data_model.drop('avg_return', axis=1, inplace=True)
X = sm.add_constant(data_model.drop('sd_return', axis=1))
y = data_model.sd_return

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=True, time_effects=True)
re_res = mod.fit()
print(re_res)

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=True, time_effects=False)
re_res = mod.fit()
print(re_res)

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=False, time_effects=True)
re_res = mod.fit()
print(re_res)

###########################
# models for predicting avg_return (only CEO vars)

data_model = data_joined.set_index(['GVKEY', 'fyear']).copy(deep=True)
data_model = data_model[
    ['AGE', 'ceo_tenure', 'dummy_chairman_president', 'dummy_chairman', 'dummy_president', 'dummy_founder',
     'ceo_group', 'avg_return', 'sd_return']]

data_model.drop('sd_return', axis=1, inplace=True)
X = sm.add_constant(data_model.drop('avg_return', axis=1))
y = data_model.avg_return

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=True, time_effects=True)
re_res = mod.fit()
print(re_res)

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=True, time_effects=False)
re_res = mod.fit()
print(re_res)

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=False, time_effects=True)
re_res = mod.fit()
print(re_res)

###########################
# models for predicting sd_return (only CEO vars)

data_model = data_joined.set_index(['GVKEY', 'fyear']).copy(deep=True)
data_model = data_model[
    ['AGE', 'ceo_tenure', 'dummy_chairman_president', 'dummy_chairman', 'dummy_president', 'dummy_founder',
     'ceo_group', 'avg_return', 'sd_return']]

data_model.drop('avg_return', axis=1, inplace=True)
X = sm.add_constant(data_model.drop('sd_return', axis=1))
y = data_model.sd_return

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=True, time_effects=True)
re_res = mod.fit()
print(re_res)

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=True, time_effects=False)
re_res = mod.fit()
print(re_res)

mod = PanelOLS(y, X, drop_absorbed=True, check_rank=False, entity_effects=False, time_effects=True)
re_res = mod.fit()
print(re_res)
