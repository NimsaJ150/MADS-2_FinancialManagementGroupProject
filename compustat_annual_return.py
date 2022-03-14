import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

"""
Load data
"""

# import data
ccm = pd.read_csv("data/CCM_Fundamentals_Annual_2006_-_2021_new.csv")
# show data
ccm.head()

ccm_return = ccm[['LPERMNO', 'fyear', 'prcc_f']]

test = ccm_return.groupby(by=['LPERMNO'])['prcc_f'].pct_change()
ccm_return_csv = pd.concat([ccm_return, test], axis=1).reset_index()
ccm_return_csv.head()
ccm_return_csv.columns = ['index', 'LPERMNO', 'fyear', 'prcc_f', 'prcc_pcchg']
ccm_return_csv.to_csv(path_or_buf='/Users/carlasuzanneweidner/Downloads/compustat_annual_return.csv', index=False)
