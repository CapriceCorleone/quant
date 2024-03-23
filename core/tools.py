'''
Author: WangXiang
Date: 2024-03-21 20:42:41
LastEditTime: 2024-03-23 22:52:30
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from numba import njit

from .aligner import Aligner
from .calendar import Calendar


def format_unstack_table(data: pd.DataFrame) -> pd.DataFrame:
    if np.issubdtype(data.columns.dtype, np.object_):
        data = data.loc[:, data.columns.str[0].str.isdigit()]
        data.columns = data.columns.str[:6].astype(int)
    if np.issubdtype(data.index.dtype, np.object_):
        data.index = data.index.astype(int)
    data.index.name, data.columns.name = 'trade_date', 'ticker'
    return data


def winsorize_mad(data: pd.DataFrame) -> pd.DataFrame:
    median = data.median(axis=1).values
    mad = np.abs(data - median[:, None]).median(axis=1).values
    ix = (median == 0) & (mad == 0)
    if ix.any():
        # 0值太多，更改mad
        data_ = data.copy()
        data_[data == 0] = np.nan
        mad_ = np.abs(data - median[:, None]).median(axis=1).values
        mad[ix] = mad_[ix]
    lower = median - 3 * 1.483 * mad
    upper = median + 3 * 1.483 * mad
    data = data.clip(lower, upper, axis=0)
    return data


def stdd_zscore(data: pd.DataFrame) -> pd.DataFrame:
    return (data - data.mean(axis=1).values[:, None]) / (data.std(axis=1).values[:, None] + 1e-8)


def __orthogonalize(y, X, universe, weight, do_orth):
    ouptut = np.zeros(y.shape) * np.nan
    for i in tqdm(range(len(y)), ncols=80, desc='orth'):
        if do_orth[i]:
            yy = y[i]
            XX = X[i]
            univ = universe[i]
            w = weight[i]
            mask = np.isnan(univ) | np.isnan(w) | (univ == 0)
            if len(yy) - XX.shape[1] - mask.sum() <= 2:
                continue
            w = w[~mask]
            w = w / w.sum()
            yy = yy[~mask]
            XX = XX[~mask]
            md = sm.WLS(yy, XX, weights=w).fit()
            resid = md.resid
            ouptut[i, ~mask] = resid
    return ouptut


def orthogonalize(y: pd.DataFrame, *X: pd.DataFrame, universe: pd.DataFrame = None, weight: pd.DataFrame = None) -> pd.DataFrame:
    aligner = Aligner()
    y = aligner.align(y).values
    X = np.stack([aligner.align(i).values for i in X]).transpose(1, 2, 0)
    if universe is None:
        universe = np.ones(y.shape)
    else:
        universe = aligner.align(universe).values
    universe[(np.isnan(y)) | (np.isnan(X).any(axis=-1))] = np.nan
    if weight is None:
        weight = np.ones(y.shape)
    else:
        weight = aligner.align(weight).values
    do_orth = np.ones(len(y))
    output = __orthogonalize(y, X, universe, weight, do_orth)
    return pd.DataFrame(output, index=aligner.trade_dates, columns=aligner.tickers)


def orthogonalize_monthend(y: pd.DataFrame, *X: pd.DataFrame, universe: pd.DataFrame = None, weight: pd.DataFrame = None) -> pd.DataFrame:
    aligner = Aligner()
    y = aligner.align(y).values
    X = np.stack([aligner.align(i).values for i in X]).transpose(1, 2, 0)
    if universe is None:
        universe = np.ones(y.shape)
    else:
        universe = aligner.align(universe).values
    universe[(np.isnan(y)) | (np.isnan(X).any(axis=-1))] = np.nan
    if weight is None:
        weight = np.ones(y.shape)
    else:
        weight = aligner.align(weight).values
    calendar = Calendar()
    do_orth = calendar.is_month_end
    output = __orthogonalize(y, X, universe, weight, do_orth)
    return pd.DataFrame(output, index=aligner.trade_dates, columns=aligner.tickers)


def orthogonalize_weekend(y: pd.DataFrame, *X: pd.DataFrame, universe: pd.DataFrame = None, weight: pd.DataFrame = None) -> pd.DataFrame:
    aligner = Aligner()
    y = aligner.align(y).values
    X = np.stack([aligner.align(i).values for i in X]).transpose(1, 2, 0)
    if universe is None:
        universe = np.ones(y.shape)
    else:
        universe = aligner.align(universe).values
    universe[(np.isnan(y)) | (np.isnan(X).any(axis=-1))] = np.nan
    if weight is None:
        weight = np.ones(y.shape)
    else:
        weight = aligner.align(weight).values
    calendar = Calendar()
    do_orth = calendar.is_week_end
    output = __orthogonalize(y, X, universe, weight, do_orth)
    return pd.DataFrame(output, index=aligner.trade_dates, columns=aligner.tickers)
