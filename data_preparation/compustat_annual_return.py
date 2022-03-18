import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

"""
Load data
"""

# import data
ccm = pd.read_csv(
    "/Users/mythulam/Desktop/Masters/02_Spring_2022/03_Guided_Studies_in_Financial_Management/Group_Project/Data/Compustat/Compustat_PRCCF_AJEX.csv")
# show data
ccm.head()
ccm['adjust_prccf'] = ccm['prcc_f'] / ccm['ajex']
ccm_return = ccm[['LPERMNO', 'fyear', 'prcc_f', 'ajex', 'adjust_prccf']]

test = ccm_return.groupby(by=['LPERMNO'])['adjust_prccf'].pct_change().where(
    ccm_return.groupby(['LPERMNO'])['fyear'].diff() == 1)
ccm_return_csv = pd.concat([ccm_return, test], axis=1).reset_index()
ccm_return_csv.head()
ccm_return_csv.columns = ['index', 'LPERMNO', 'fyear', 'prcc_f', 'ajex', 'adjust_prccf', 'prcc_pcchg']
ccm_return_csv.to_csv(
    path_or_buf='/Users/mythulam/Desktop/Masters/02_Spring_2022/03_Guided_Studies_in_Financial_Management/Group_Project/Data/Compustat/Compustat_annual_return_NEW.csv',
    index=False)

ccm['adjust_prccf'] = ccm['adjust_prccf'] / (ccm['ajex'] * ccm['ajex'])
ccm = ccm.loc[ccm['prcc_pcchg'] < 50]
ccm.to_csv(path_or_buf='data/Compustat_annual_return.csv', index=False)
