import numpy as np
import pandas as pd

"""
Load data
"""

# import data
sec_daily_all = pd.read_csv(
    "/Users/mythulam/Desktop/Masters/02_Spring_2022/03_Guided_Studies_in_Financial_Management/Group_Project/Data/CRSP/security_all.csv")

# show data
sec_daily_all.head()
sec_daily_all.info()        # MyThu added
# 23,339,902 records        # MyThu added

sec_daily_all[['LPERMNO', 'datadate']].head()
# data is not sorted by date

print(sec_daily_all.isnull().sum())         # MyThu added
# prccd - 245 null values (0.001%)          # MyThu added
# ajexdi - 245 null values (0.001%)         # MyThu added
# trfd - 6,325,394 null values (27.10%)     # MyThu added

"""
Preprocess Data
"""

# drop rows with nan in prccd, ajexdi, and trfd columns or 0 in ajexdi column
sec_daily = sec_daily_all
sec_daily = sec_daily[sec_daily['prccd'].notna()]
sec_daily = sec_daily[sec_daily['ajexdi'].notna()]
sec_daily = sec_daily[sec_daily['trfd'].notna()]
sec_daily = sec_daily.loc[~(sec_daily['ajexdi'] == 0)]
sec_daily = sec_daily.reset_index()

sec_daily.info()        # MyThu added
# 16,989,230 records    # MyThu added
# 72.79%                # MyThu added

"""
Process Data
"""

# create year field
sec_daily['year'] = sec_daily['datadate'].astype(str).str[0:4]

# calculate adjusted closing price
sec_daily['adj_prccd'] = (sec_daily['prccd'] / sec_daily['ajexdi']) * sec_daily['trfd']

# sort data by datadate
sec_daily = sec_daily.sort_values(
    ['LPERMNO', 'datadate'], ascending=[True, True])
sec_daily[['LPERMNO', 'datadate']].head()

# create shifted column
sec_daily['prev_adj_prccd'] = sec_daily.groupby(by=['LPERMNO'])[
    'adj_prccd'].shift(1)
sec_daily[['LPERMNO', 'datadate', 'adj_prccd', 'prev_adj_prccd']].head()

# calculate daily return
sec_daily['return'] = sec_daily['adj_prccd'] / sec_daily['prev_adj_prccd'] - 1
sec_daily[['LPERMNO', 'datadate', 'adj_prccd',
           'prev_adj_prccd', 'return']].head()

# calculate standard deviation per LPERMNO per year
return_sd = sec_daily.groupby(by=['LPERMNO', 'year'])['return'].std()
return_sd.head()

# create new CSV with results
return_sd.to_csv(
    'data/CRSP_annual_standard_deviation.csv')
