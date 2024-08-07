'''
Author: WangXiang
Date: 2024-03-30 15:21:47
LastEditTime: 2024-04-14 01:44:18
'''

import numpy as np
import pandas as pd

from ..core import Processors
from ..core.njit.financial import *
from ..core.njit.analyst import *


def unstack_market_cap(AShareEODDerivativeIndicator: pd.DataFrame, init_date: int) -> pd.DataFrame:
    df_market_cap = AShareEODDerivativeIndicator.loc[str(init_date):, 'S_VAL_MV'].unstack() * 1.0e4
    df_market_cap.index = df_market_cap.index.astype(int)
    return df_market_cap

def unstack_float_shares(AShareEODDerivativeIndicator: pd.DataFrame, init_date: int) -> pd.DataFrame:
    # df_float_shares = AShareEODDerivativeIndicator.loc[str(init_date):, 'FLOAT_A_SHR_TODAY'].unstack() * 1.0e4
    df_float_shares = AShareEODDerivativeIndicator.loc[str(init_date):, 'TOT_SHR_TODAY'].unstack() * 1.0e4
    df_float_shares.index = df_float_shares.index.astype(int)
    return df_float_shares



# %% 估值因子
def bp(AShareEODDerivativeIndicator: pd.DataFrame, AShareBalanceSheet: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(x)
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareBalanceSheet], ['TOT_SHRHLDR_EQY_EXCL_MIN_INT'], operator, init_date)
    return factor


def ep_ttm(AShareEODDerivativeIndicator: pd.DataFrame, AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_ttm(x))
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareIncome], ['NET_PROFIT_EXCL_MIN_INT_INC'], operator, init_date)
    return factor


def ep_mrq(AShareEODDerivativeIndicator: pd.DataFrame, AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_mrq(x))
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareIncome], ['NET_PROFIT_EXCL_MIN_INT_INC'], operator, init_date)
    return factor


def sp_ttm(AShareEODDerivativeIndicator: pd.DataFrame, AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_ttm(x))
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareIncome], ['OPER_REV'], operator, init_date)
    return factor


def sp_mrq(AShareEODDerivativeIndicator: pd.DataFrame, AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_mrq(x))
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareIncome], ['OPER_REV'], operator, init_date)
    return factor


def op_ttm(AShareEODDerivativeIndicator: pd.DataFrame, AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_ttm(x))
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareIncome], ['OPER_PROFIT'], operator, init_date)
    return factor


def op_mrq(AShareEODDerivativeIndicator: pd.DataFrame, AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_mrq(x))
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareIncome], ['OPER_PROFIT'], operator, init_date)
    return factor


def ocfp_ttm(AShareEODDerivativeIndicator: pd.DataFrame, AShareCashFlow: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_ttm(x))
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareCashFlow], ['CASH_RECP_SG_AND_RS'], operator, init_date)
    return factor


def ocfp_mrq(AShareEODDerivativeIndicator: pd.DataFrame, AShareCashFlow: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_mrq(x))
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareCashFlow], ['CASH_RECP_SG_AND_RS'], operator, init_date)
    return factor


"""
TODO:
dp_ttm
rndp_ttm
ep_fttm
ep_hp_filtered
fcfp_ttm
fcfp_mrq
"""


# %% 成长因子
def np_mrq_yoy(AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_yoy(ffunc_mrq(x)))
    factor = Processors.fundamental.process([AShareIncome], ['NET_PROFIT_EXCL_MIN_INT_INC'], operator, init_date)
    return factor


def np_ttm_yoy(AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_yoy(ffunc_ttm(x)))
    factor = Processors.fundamental.process([AShareIncome], ['NET_PROFIT_EXCL_MIN_INT_INC'], operator, init_date, align_date=kwargs.get('align_date', True))
    return factor


def rev_mrq_yoy(AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_yoy(ffunc_mrq(x)))
    factor = Processors.fundamental.process([AShareIncome], ['OPER_REV'], operator, init_date)
    return factor


def rev_ttm_yoy(AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_yoy(ffunc_ttm(x)))
    factor = Processors.fundamental.process([AShareIncome], ['OPER_REV'], operator, init_date, align_date=kwargs.get('align_date', True))
    return factor


def op_mrq_yoy(AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_yoy(ffunc_mrq(x)))
    factor = Processors.fundamental.process([AShareIncome], ['OPER_PROFIT'], operator, init_date)
    return factor


def op_ttm_yoy(AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_yoy(ffunc_ttm(x)))
    factor = Processors.fundamental.process([AShareIncome], ['OPER_PROFIT'], operator, init_date)
    return factor

# by pyq （不一定对）
def roe_mrq(AShareFinancialIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(x)
    factor = Processors.fundamental.process([AShareFinancialIndicator], ['S_QFA_ROE'], operator, init_date)
    return factor

def roe_ttm(AShareIncome: pd.DataFrame, AShareBalanceSheet: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x, y):
        return ffunc_last(ffunc_divide(ffunc_ttm(x), ffunc_mean(y, 4)))
    factor = Processors.fundamental.process([AShareIncome, AShareBalanceSheet], ['NET_PROFIT_EXCL_MIN_INT_INC', 'TOT_SHRHLDR_EQY_EXCL_MIN_INT'], operator, init_date)
    return factor

def roe_mrq_yoy(AShareFinancialIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_yoy(x, method='diff'))
    factor = Processors.fundamental.process([AShareFinancialIndicator], ['S_QFA_ROE'], operator, init_date)
    return factor

def roe_ttm_yoy(AShareIncome: pd.DataFrame, AShareBalanceSheet: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x, y):
        return ffunc_last(ffunc_yoy(ffunc_divide(ffunc_ttm(x), ffunc_mean(y, 4)), method='diff'))
    factor = Processors.fundamental.process([AShareIncome, AShareBalanceSheet], ['NET_PROFIT_EXCL_MIN_INT_INC', 'TOT_SHRHLDR_EQY_EXCL_MIN_INT'], operator, init_date)
    return factor

def eps_mrq(AShareFinancialIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(x)
    factor = Processors.fundamental.process([AShareFinancialIndicator], ['S_QFA_EPS'], operator, init_date)  # .fundamental用来处理只涉及财务数据的因子
    return factor

# def eps_ttm(AShareIncome: pd.DataFrame, AShareEODDerivativeIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
#     def operator(x): # 先把财务数据处理成日频
#         return ffunc_last((ffunc_ttm(x))
#     shares = unstack_float_shares(AShareEODDerivativeIndicator, init_date) # shares是日频
#     factor = Processors.value.process(shares, [AShareIncome], ['NET_PROFIT_EXCL_MIN_INT_INC'], operator, init_date)  # .value用来处理只涉及财务数据的因子除以一个日频数据
#     return factor 

def eps_ttm(AShareIncome: pd.DataFrame, AShareEODDerivativeIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    shares = (AShareEODDerivativeIndicator.loc(axis=0)[str(init_date - 50000):, ]['TOT_SHR_TODAY'] * 10000).reset_index()
    income = pd.merge(AShareIncome[['S_INFO_WINDCODE', 'ANN_DT', 'REPORT_PERIOD', 'NET_PROFIT_EXCL_MIN_INT_INC']],
                      shares,
                      left_on=['S_INFO_WINDCODE', 'ANN_DT'],
                      right_on=['S_INFO_WINDCODE', 'TRADE_DT'])
    def operator(x, y):
        return ffunc_last(ffunc_divide(ffunc_ttm(x), ffunc_mean(y)))
    factor = Processors.fundamental.process([income, income], ['NET_PROFIT_EXCL_MIN_INT_INC', 'TOT_SHR_TODAY'], operator, init_date)
    return factor

def eps_mrq_yoy(AShareFinancialIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_yoy(x))
    factor = Processors.fundamental.process([AShareFinancialIndicator], ['S_QFA_EPS'], operator, init_date)
    return factor

# def eps_ttm_yoy(AShareIncome: pd.DataFrame, AShareEODDerivativeIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
#     def operator(x): # 先把财务数据处理成日频
#         return ffunc_last(ffunc_yoy(ffunc_ttm(x)))
#     shares = unstack_float_shares(AShareEODDerivativeIndicator, init_date) # shares是日频
#     factor = Processors.value.process(shares, [AShareIncome], ['NET_PROFIT_EXCL_MIN_INT_INC'], operator, init_date)  # .value用来处理只涉及财务数据的因子除以一个日频数据
#     return factor 

def eps_ttm_yoy(AShareIncome: pd.DataFrame, AShareEODDerivativeIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    shares = (AShareEODDerivativeIndicator.loc(axis=0)[str(init_date - 50000):, ]['TOT_SHR_TODAY'] * 10000).reset_index()
    income = pd.merge(AShareIncome[['S_INFO_WINDCODE', 'ANN_DT', 'REPORT_PERIOD', 'NET_PROFIT_EXCL_MIN_INT_INC']],
                      shares,
                      left_on=['S_INFO_WINDCODE', 'ANN_DT'],
                      right_on=['S_INFO_WINDCODE', 'TRADE_DT'])
    def operator(x, y):
        return ffunc_last(ffunc_yoy(ffunc_divide(ffunc_ttm(x), ffunc_mean(y))))
    factor = Processors.fundamental.process([income, income], ['NET_PROFIT_EXCL_MIN_INT_INC', 'TOT_SHR_TODAY'], operator, init_date)
    return factor 

def eps_ttm_qoq(AShareIncome: pd.DataFrame, AShareEODDerivativeIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    shares = (AShareEODDerivativeIndicator.loc(axis=0)[str(init_date - 50000):, ]['TOT_SHR_TODAY'] * 10000).reset_index()
    income = pd.merge(AShareIncome[['S_INFO_WINDCODE', 'ANN_DT', 'REPORT_PERIOD', 'NET_PROFIT_EXCL_MIN_INT_INC']],
                      shares,
                      left_on=['S_INFO_WINDCODE', 'ANN_DT'],
                      right_on=['S_INFO_WINDCODE', 'TRADE_DT'])
    def operator(x, y):
        return ffunc_last(ffunc_qoq(ffunc_divide(ffunc_ttm(x), ffunc_mean(y, 4))))
    factor = Processors.fundamental.process([income, income], ['NET_PROFIT_EXCL_MIN_INT_INC', 'TOT_SHR_TODAY'], operator, init_date)
    return factor 




"""
TODO:

adv_yoy
adv_yoy_liab
np_deduct_mrq_yoy

np_mrq_trend
np_mrq_yoy_mscore
np_mrq_yoy_zscore
np_mrq_zscore
np_mrq_zscore_yoy
np_mrq_zscore_yoy_qoq
np_ttm_qoq

pead_open
pead_low

rev_mrq_trend
rev_mrq_yoy_mscore
rev_mrq_yoy_zscore
rev_mrq_zscore
rev_mrq_zscore_yoy
rev_mrq_zscore_yoy_qoq
rev_ttm_qoq

roe_mrq_yoy
roe_mrq_trend
roe_mrq_yoy_mscore
roe_mrq_zscore
roe_mrq_zscore_yoy
roe_mrq_zscore_yoy_qoq
roe_ttm_qoq

sue
sui

peg_fy0
peg_fy1
"""


# %% 质量因子


"""
TODO:
adv2pre
asset_turn_mrq
asset_turn_ttm
pay2receive
rnd2asset
rnd2cost
rnd2equity
rnd2rev
roe_mrq
roe_ttm
roic
shareholder_num_zscore
"""