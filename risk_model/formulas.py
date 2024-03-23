'''
Author: WangXiang
Date: 2024-03-23 21:28:39
LastEditTime: 2024-03-24 01:53:15
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from numba import njit
from scipy import stats

from ..core import Aligner, Calendar, format_unstack_table


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
    for i in tqdm(range(len(stock_return))):
        if i < window - 1:
            continue
        stock_rtn = stock_return[i - window : i + 1]
        index_rtn = index_return[i - window : i + 1]
        index_rtn_isnan = np.isnan(index_rtn)
        for j in tqdm(range(stock_return.shape[1]), desc=i):
            if universe[i, j] == 0:
                continue
            y = stock_rtn[j]
            mask = np.isnan(y) | index_rtn_isnan
            if window - mask.sum() < min_periods:
                continue
            y = y[~mask]
            x = index_rtn[~mask]
            x_ = np.append(x, np.ones((len(y), 1), dtype=x.dtype), axis=1)
            x_T = x_.T
            w = np.diag(weight_decay[~mask])
            beta = np.linalg.inv(x_T @ w @ x_) @ x_T @ w @ y
            yhat = x_ @ beta
            s2 = np.sum((yhat - y) ** 2) / (len(y) - 2)
            x_var = np.sum((x - x.mean()) ** 2)
            se2 = s2 / x_var
            output_beta[i, j] = beta
            output_se2[i, j] = se2
    return output_beta, output_se2


def beta_sub(AShareEODPrices: pd.DataFrame, AIndexEODPrices: pd.DataFrame, init_date: int, num_process: int = 1, **kwargs):
    """
    WLS回归，然后按照行业贝叶斯压缩，先验值为行业内市值加权beta，后验值为回归beta，系数由回归beta的标准误和行业内beta的方差计算得到，详见
    http://diskussionspapiere.wiwi.uni-hannover.de/pdf_bib/dp-617.pdf
    式2
    """
    window = kwargs['window']
    min_periods = kwargs['min_periods']
    half_life = kwargs['half_life']
    ratio = kwargs['ratio']
    universe = kwargs['universe'].values
    weight = kwargs['weight']
    aligner = Aligner()
    calendar = Calendar()
    start_date = calendar.get_prev_trade_date(init_date, window - 1)
    stock_quote = AShareEODPrices.copy()
    stock_quote.loc[stock_quote['S_DQ_TRADE_STATUSCODE'] == 0, 'S_DQ_PCTCHANGE'] = np.nan
    stock_return = aligner.align(format_unstack_table(stock_quote['S_DQ_PCTCHANGE'].unstack())).loc[start_date:]
    index = stock_return.index
    columns = stock_return.columns
    stock_return = stock_return.values.astype(np.float32)
    index_return = AIndexEODPrices.loc(axis=0)[:, '000985.CSI']['S_DQ_PCTCHANGE']
    index_return.index = index_return.index.astype(int)
    index_return = index_return.reindex(index=aligner.trade_dates).loc[start_date:].values.astype(np.float32)
    weight_decay = exponential_weight(window, half_life, stock_return.dtype, ratio)
    output_beta, output_se2 = beta_sub_ufunc(stock_return, index_return, window, min_periods, weight_decay, universe)
    output_beta = pd.DataFrame(output_beta, index=index, columns=columns)
    output_se2 = pd.DataFrame(output_se2, index=index, columns=columns)
    return output_beta