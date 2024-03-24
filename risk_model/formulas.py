'''
Author: WangXiang
Date: 2024-03-23 21:28:39
LastEditTime: 2024-03-24 13:57:10
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from numba import njit
from scipy import stats

from ..core import Aligner, Calendar, Universe, format_unstack_table


def exponential_weight(seq_len: int, half_life: int, dtype: np.dtype, ratio: float = 0.5):
    # starts with 1, e.g., [1, 1/2, 1/4, 1/8, ...]
    # output: 1-d array
    ratio = np.power(ratio, 1 / half_life) if half_life is None else 1
    weights = np.array([np.power(ratio, i) for i in range(seq_len)], dtype=dtype)
    return weights


# %% Size
def size_sub(AShareEODDerivativeIndicator: pd.DataFrame, init_date: int, **kwargs):
    factor = AShareEODDerivativeIndicator.loc[str(init_date):, 'S_VAL_MV'].unstack()
    factor = np.log(factor + 1)
    return factor


# %% Beta
# @njit
def beta_sub_ufunc(stock_return, index_return, window, min_periods, weight_decay, universe):
    output_beta = np.zeros(stock_return.shape) * np.nan
    output_se2 = np.zeros(stock_return.shape) * np.nan
    for i in range(len(stock_return)):
        if i < window - 1:
            continue
        stock_rtn = stock_return[i - window + 1 : i + 1]
        index_rtn = index_return[i - window + 1 : i + 1]
        index_rtn_isnan = np.isnan(index_rtn)
        for j in range(stock_return.shape[1]):
            if universe[i, j] == 0:
                continue
            y = stock_rtn[:, j]
            mask = np.isnan(y) | index_rtn_isnan
            if window - mask.sum() < min_periods:
                continue
            y = y[~mask]
            x = index_rtn[~mask]
            x_ = np.append(x[:, None], np.ones((len(y), 1), dtype=x.dtype), axis=1)
            x_T = x_.T
            w = np.diag(weight_decay[~mask])
            beta = np.linalg.inv(x_T @ w @ x_) @ x_T @ w @ y
            yhat = x_ @ beta
            s2 = np.sum((yhat - y) ** 2) / (len(y) - 2)
            x_var = np.sum((x - x.mean()) ** 2)
            se2 = s2 / x_var
            output_beta[i, j] = beta[0]
            output_se2[i, j] = se2
    return output_beta, output_se2


@njit
def beta_shrinkage(beta, se2, industry, weight):
    factor = np.zeros(beta.shape) * np.nan
    for i in range(len(beta)):
        b = beta[i]
        s = se2[i]
        d = industry[i]
        w = weight[i]
        mask = np.isnan(b) | np.isnan(s) | (d == 0) | np.isnan(w)
        if mask.sum() == len(b):
            continue
        bb = b[~mask]
        ss = s[~mask]
        dd = d[~mask]
        ww = w[~mask]
        dd_max = dd.max()
        ff = np.zeros(len(bb)) * np.nan
        for j in range(1, dd_max + 1):
            ix = dd == j
            if ix.sum() == 0:
                continue
            bbb = bb[ix]
            sss = ss[ix]
            www = ww[ix]
            mean = (bbb * www).sum() / www.sum()
            var = ((bbb - mean) ** 2).sum() / len(bbb)
            coef = var / (var + sss)
            ff[ix] = coef * bbb + (1 - coef) * mean
        factor[i, ~mask] = ff
    return factor


def beta_sub(AShareEODPrices: pd.DataFrame, AIndexEODPrices: pd.DataFrame, AShareIndustriesClassCITICS: pd.DataFrame, init_date: int, num_process: int = 1, **kwargs):
    """
    WLS回归，然后按照行业贝叶斯压缩，先验值为行业内市值加权beta，后验值为回归beta，系数由回归beta的标准误和行业内beta的方差计算得到，详见
    http://diskussionspapiere.wiwi.uni-hannover.de/pdf_bib/dp-617.pdf
    式2
    """
    window = kwargs['window']
    min_periods = kwargs['min_periods']
    half_life = kwargs['half_life']
    ratio = kwargs['ratio']
    universe = kwargs['universe']
    weight = kwargs['weight'].values
    aligner = Aligner()
    calendar = Calendar()
    
    start_date = calendar.get_prev_trade_date(init_date, window - 1)
    stock_quote = AShareEODPrices.copy()
    stock_quote.loc[stock_quote['S_DQ_TRADESTATUSCODE'] == 0, 'S_DQ_PCTCHANGE'] = np.nan
    stock_return = aligner.align(format_unstack_table(stock_quote['S_DQ_PCTCHANGE'].unstack())).loc[start_date:]
    index = stock_return.index
    columns = stock_return.columns
    stock_return = stock_return.values.astype(np.float32)
    index_return = AIndexEODPrices.loc(axis=0)[:, '000985.CSI']['S_DQ_PCTCHANGE'].droplevel(1)
    index_return.index = index_return.index.astype(int)
    index_return = index_return.reindex(index=aligner.trade_dates).loc[start_date:].values.astype(np.float32)
    weight_decay = exponential_weight(window, half_life, stock_return.dtype, ratio)
    output_beta, output_se2 = beta_sub_ufunc(stock_return, index_return, window, min_periods, weight_decay, universe.loc[start_date:].values)
    output_beta = aligner.align(pd.DataFrame(output_beta, index=index, columns=columns)).values
    output_se2 = aligner.align(pd.DataFrame(output_se2, index=index, columns=columns)).values

    univ = Universe()
    industry = np.zeros(aligner.shape, dtype=int)
    indnames = AShareIndustriesClassCITICS['INDUSTRIESNAME'].unique()
    for i, name in enumerate(indnames):
        info = AShareIndustriesClassCITICS[AShareIndustriesClassCITICS['INDUSTRIESNAME'] == name][['S_INFO_WINDCODE', 'ENTRY_DT', 'REMOVE_DT']]
        industry += univ._format_universe(univ.arrange_info_table(info)).fillna(value=0).values.astype(int) * (i + 1)
    factor = beta_shrinkage(output_beta, output_se2, industry, weight)
    factor = pd.DataFrame(factor, index=aligner.trade_dates, columns=aligner.tickers)
    factor = aligner.align(factor).loc[init_date:]
    return factor