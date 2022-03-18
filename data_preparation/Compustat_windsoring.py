from cmath import nan
from pickle import FALSE
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

data = pd.read_csv(
    '/Users/mythulam/Desktop/Masters/02_Spring_2022/03_Guided_Studies_in_Financial_Management/Group_Project/Data/Compustat/CCM_Fundamentals_Annual_2006_-_2021_NEW.csv')

# Company Features - ROA, Tobins_Q, Cash_Flow, Leverage, Investment, Cash_Holdings, Div_over_Earn, aqc

"""
FIX LEVERAGE AND ACQUISITION VARIABLES
"""


# fix leverage


def leverage(row):
    return (row['dltt'] + row['dlc']) / row['ceq'] if row['ceq'] != 0 else 0


data['Leverage'] = data.apply(lambda row: leverage(row), axis=1)

# check leverage
data[['dltt', 'dlc', 'ceq', 'Leverage']].head()


# fix acquisitions to number of acquisitions


def acquisitions(row):
    return 0 if row['aqc'] == 0 or pd.isna(row['aqc']) == True else 1


data['Acquisitions'] = data.apply(lambda row: acquisitions(row), axis=1)

"""
UNDERSTAND DATASET
"""

# Data Summary Statistics
print(data.isnull().sum())

data[["ROA"]].describe()
data[["Tobins_Q"]].describe()
data[["Cash_Flow"]].describe()
data[["Leverage"]].describe()
data[["Investment"]].describe()
data[["Cash_Holdings"]].describe()
data[["Div_over_Earn"]].describe()
data[["aqc"]].describe()

"""
DETERMINE WINSOR QUANTILES
"""


# outer fences of variables


def fences(df, variable_name):
    q1 = df[variable_name].quantile(0.25)
    q3 = df[variable_name].quantile(0.75)
    iqr = q3 - q1
    outer_fence = 3 * iqr
    outer_fence_le = q1 - outer_fence
    outer_fence_ue = q3 + outer_fence
    print(outer_fence_le)
    print(outer_fence_ue)
    return outer_fence_le, outer_fence_ue


# quantile values


def quantiles(df, variable_name):
    print(df[variable_name].quantile(0.001))
    print(df[variable_name].quantile(0.01))
    print(df[variable_name].quantile(0.05))
    print(df[variable_name].quantile(0.95))
    print(df[variable_name].quantile(0.99))
    print(df[variable_name].quantile(0.999))


# ROA
outer_fence_le, outer_fence_ue = fences(data, 'ROA')
quantiles(data, 'ROA')
# Winsor Quantile Chosen: 1%, 99%

# Tobins_Q
outer_fence_le, outer_fence_ue = fences(data, 'Tobins_Q')
quantiles(data, 'Tobins_Q')
# Winsor Quantile Chosen: 95%

# Cash_Flow
outer_fence_le, outer_fence_ue = fences(data, 'Cash_Flow')
quantiles(data, 'Cash_Flow')
# Winsor Quantile Chosen: 1%, 99%

# Leverage
outer_fence_le, outer_fence_ue = fences(data, 'Leverage')
quantiles(data, 'Leverage')
# Winsor Quantile Chosen: 5%, 95%

# Investment
outer_fence_le, outer_fence_ue = fences(data, 'Investment')
quantiles(data, 'Investment')
# Winsor Quantile Chosen: 1%, 99%

# Cash_Holdings
outer_fence_le, outer_fence_ue = fences(data, 'Cash_Holdings')
quantiles(data, 'Cash_Holdings')
# Winsor Quantile Chosen: 95%

# Div_Over_Earn
outer_fence_le, outer_fence_ue = fences(data, 'Div_over_Earn')
quantiles(data, 'Div_over_Earn')
# Winsor Quantile Chosen: 95%

"""
WINSORIZE DATA
"""
# create copy of data to winsor
data_win = data.copy(deep=True)

"""
ISSUES WITH THIS WINSORIZE FUNCTION AND NAN VALUES - DO NOT USE
"""
# winsorize ROA
data_win['ROA_w01_w99'] = winsorize(data['ROA'], limits=(0.01, 0.01))
data_win['ROA_w01_w99'].describe()

# winsorize Tobins_Q
data_win['Tobins_Q_w95'] = winsorize(data['Tobins_Q'], limits=(0, 0.05))
data_win['Tobins_Q_w95'].describe()

# winsorize Cash_Flow
# WE'VE GOT PROBLEMS WITH THIS ONE
data_win['CF_w01_w99'] = winsorize(data['Cash_Flow'], limits=(0.01, 0.01))
data_win['CF_w01_w99'].describe()

# Leverage
data_win['Lev_w05_w95'] = winsorize(data['Leverage'], limits=(0.05, 0.05))
data_win['Lev_w05_w95'].describe()

# Investment
# WE'VE GOT PROBLEMS WITH THIS ONE
data_win['Inv_w01_w99'] = winsorize(data['Investment'], limits=(0.01, 0.01))
data_win['Inv_w01_w99'].describe()

# Cash_Holdings
data_win['CashHold_w95'] = winsorize(data['Cash_Holdings'], limits=(0, 0.05))
data_win['CashHold_w95'].describe()

# Div_over_Earn
# WE'VE GOT PROBLEMS WITH THIS ONE
data_win['DoE_w01_w99'] = winsorize(data['Div_over_Earn'], limits=(0.01, 0.01))
data_win['DoE_w01_w99'].describe()

"""
USE THIS WINSORIZE INSTEAD
"""


def winsorize_with_pandas(s, limits):
    """
    s : pd.Series
    Series to winsorize
    limits : tuple of float
    Tuple of the percentages to cut on each side of the array, 
    with respect to the number of unmasked data, as floats between 0. and 1
    """
    return s.clip(lower=s.quantile(limits[0], interpolation='lower'),
                  upper=s.quantile(1 - limits[1], interpolation='higher'))


# winsorize ROA
data_win['ROA_w01_w99'] = winsorize_with_pandas(
    data['ROA'], limits=(0.01, 0.01))
data_win['ROA_w01_w99'].describe()

# winsorize Tobins_Q
data_win['Tobins_Q_w95'] = winsorize_with_pandas(
    data['Tobins_Q'], limits=(0, 0.05))
data_win['Tobins_Q_w95'].describe()

# winsorize Cash_Flow
data_win['CF_w01_w99'] = winsorize_with_pandas(
    data['Cash_Flow'], limits=(0.01, 0.01))
data_win['CF_w01_w99'].describe()

# Leverage
data_win['Lev_w05_w95'] = winsorize_with_pandas(
    data['Leverage'], limits=(0.05, 0.05))
data_win['Lev_w05_w95'].describe()

# Investment
data_win['Inv_w01_w99'] = winsorize_with_pandas(
    data['Investment'], limits=(0.01, 0.01))
data_win['Inv_w01_w99'].describe()

# Cash_Holdings
data_win['CashHold_w95'] = winsorize_with_pandas(
    data['Cash_Holdings'], limits=(0, 0.05))
data_win['CashHold_w95'].describe()

# Div_over_Earn
data_win['DoE_w01_w99'] = winsorize_with_pandas(
    data['Div_over_Earn'], limits=(0.01, 0.01))
data_win['DoE_w01_w99'].describe()

# check data
print(data_win.head())
print(data_win.info())

data_win.to_csv(
    'data/CCM_Fundamentals_Annual_2006_-_2021_winsorized.csv',
    index=False)
