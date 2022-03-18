import pandas as pd
import numpy as np


# new columns
# ROA = NI / AT
# tobins Q = (AT + (CSHO * PRCC_F) - CEQ) / AT
# Cash Flow = (IBC + DP) / AT
# Leverage = (DLTT + DLC) / CEQ
# Investment = CAPX / PPENT
# Cash Holdings = CHE/PPENT
# Dividends over earnings = (DVC + DVP) / EBITDA
# R&D = RDIP / AT (of the past year)
# SG&A = XSGA / SALE
# Return on Assets = EBITDA / AT (of the past year)
# Operating Return = OANCF / AT (of the past year)
# Acquisitions = AQC
# Interest Coverage = EBITDA / TIE

def roa(row):
    return row['ni'] / row['at'] if row['at'] != 0 else 0


def tobins_q(row):
    return (row['at'] + (row['csho'] * row['prcc_f']) - row['ceq']) / row['at'] if row['at'] != 0 else 0


def cash_flow(row):
    return (row['ibc'] + row['dp']) / row['at'] if row['at'] != 0 else 0


def leverage(row):
    return row['dltt'] + row['dlc'] / row['ceq'] if row['ceq'] != 0 else 0


def investment(row):
    return row['capx'] / row['ppent'] if row['ppent'] != 0 else 0


def cash_holdings(row):
    return row['che'] / row['ppent'] if row['ppent'] != 0 else 0


def div_over_earn(row):
    return (row['dvc'] + row['dvp']) / row['ebitda'] if row['ebitda'] != 0 else 0


def sg_a(row):
    return row['xsga'] / row['sale'] if row['sale'] != 0 else 0


# def interest_coverage(row):
#     return row['ebitda'] / row['tie'] if row['tie'] != 0 else 0

data = pd.read_csv('data_raw/CCM_Fundamentals_Annual_2006_-_2021_clipped.csv')

data['ROA'] = data.apply(lambda row: roa(row), axis=1)
data['Tobins_Q'] = data.apply(lambda row: tobins_q(row), axis=1)
data['Cash_Flow'] = data.apply(lambda row: cash_flow(row), axis=1)
data['Leverage'] = data.apply(lambda row: leverage(row), axis=1)
data['Investment'] = data.apply(lambda row: investment(row), axis=1)
data['Cash_Holdings'] = data.apply(lambda row: cash_holdings(row), axis=1)
data['Div_over_Earn'] = data.apply(lambda row: div_over_earn(row), axis=1)
data['SG_A'] = data.apply(lambda row: sg_a(row), axis=1)

data.to_csv('data/CCM_Fundamentals_Annual_2006_-_2021_normal.csv')
