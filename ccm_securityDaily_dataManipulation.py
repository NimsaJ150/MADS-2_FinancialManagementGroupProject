import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

"""
Load data
"""

# import data
sec_daily_all = pd.read_csv("/Users/carlasuzanneweidner/Downloads/security_all.csv")
# show data
sec_daily_all.head()


"""
Preprocess Data
"""

# drop nan in prcc_f or ajex/trfd columns
sec_daily = sec_daily_all[sec_daily_all['prccd'].notna()] # change to closing price
sec_daily = sec_daily[sec_daily['ajexdi'].notna()]
sec_daily = sec_daily[sec_daily['trfd'].notna()]
sec_daily = sec_daily.loc[~(sec_daily['ajexdi']==0)]
sec_daily = sec_daily.reset_index()


# group on company and sort on date
sec_daily['adj_prccd'] = (sec_daily['prccd'] / sec_daily['ajexdi']) * sec_daily['trfd']
sec_daily['year'] = sec_daily['datadate'].astype(str).str[0:4]
stdev_csv = sec_daily.groupby(by=['LPERMNO', 'year'])['adj_prccd'].std()
stdev_csv = stdev_csv.to_frame()

stdev_csv.reset_index(inplace=True)

stdev_csv.to_csv(path_or_buf='/Users/carlasuzanneweidner/Downloads/crsp_stdev_annual.csv', index=False)




#
# ### ignore from here down
# # calculate return
# numerator = (sec_daily['prccd']/sec_daily['ajexdi'])*sec_daily['trfd']
# denominator = [1]
#
#
# for i in range(1,len(numerator)):
#     den = ((sec_daily['prccd'][i-1])/(sec_daily['ajexdi'][i-1]))\
#           *(sec_daily['trfd'][i-1])
#     denominator.append(den)
# sec_daily['return'] = numerator/denominator - 1
# #(((((PRCCD/AJEXDI)*TRFD) / ((PRCCD(PRIOR)/AJEXDI(PRIOR))*TRFD(PRIOR))) -1 )* 100).
#
# # aggregate, calculate standard deviation
# # create year column
# sec_daily['year'] = sec_daily['datadate'].astype(str).str[0:4]
# unique_yr = sec_daily['year'].unique()
# # use LPERMNO to group by company
# unique_lp = sec_daily['LPERMNO'].unique()
#
# for company in unique_lp:
#     returns = sec_daily.loc[(sec_daily['LPERMNO']==unique_lp)]
#

# return_avg = sec_daily.groupby(by=['LPERMNO', 'year'])['return'].mean()
# return_sd = sec_daily.groupby(by=['LPERMNO', 'year'])['return'].std()
#
# return_avg.reset_index()
# return_sd.reset_index()
#
# return_annual = pd.concat([return_avg, return_sd], axis=1).reset_index()
# return_annual.columns = ['LPERMNO', 'year', 'avg_return', 'sd_return']
#
# return_annual.to_csv(path_or_buf='/Users/carlasuzanneweidner/Downloads/return_annual.csv', index=False)