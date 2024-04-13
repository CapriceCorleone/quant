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
    factor = Processors.fundamental.process([AShareIncome], ['NET_PROFIT_EXCL_MIN_INT_INC'], operator, init_date)
    return factor


def rev_mrq_yoy(AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_yoy(ffunc_mrq(x)))
    factor = Processors.fundamental.process([AShareIncome], ['OPER_REV'], operator, init_date)
    return factor


def rev_ttm_yoy(AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_yoy(ffunc_ttm(x)))
    factor = Processors.fundamental.process([AShareIncome], ['OPER_REV'], operator, init_date)
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