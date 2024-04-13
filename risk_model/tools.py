'''
Author: WangXiang
Date: 2024-03-23 21:28:59
LastEditTime: 2024-04-13 21:50:44
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from statsmodels.stats.stattools import medcouple


def winsorize_box(data: pd.DataFrame) -> pd.DataFrame:
    q1 = data.quantile(0.25, axis=1).values
    q3 = data.quantile(0.75, axis=1).values
    iqr = q3 - q1
    mc = np.zeros(len(data)) * np.nan
    for i in tqdm(range(len(data)), ncols=80, desc='box'):
        x = data.iloc[i].values
        x = x[~np.isnan(x)]
        if len(x) <= 10:
            q1[i] = np.nan
            q3[i] = np.nan
            iqr[i] = np.nan
            continue
        mc[i] = medcouple(x)
    lower_k = -3.5 * (mc >= 0) - 4 * (mc < 0)
    upper_k = 4 * (mc >= 0) + 3.5 * (mc < 0)
    lower = q1 - 1.5 * np.power(np.e, lower_k * mc) * iqr
    upper = q3 + 1.5 * np.power(np.e, upper_k * mc) * iqr
    data = data.clip(lower, upper, axis=0)
    return data


def weighted_stdd_zscore(data, weight):
    mask = pd.isna(data) | pd.isna(weight)
    data = data[~mask]
    weight = weight[~mask]
    mean = (data * weight).sum(axis=1) / weight.sum(axis=1)
    std = data.std(axis=1)
    data = (data - mean.values[:, None]) / (std.values[:, None] + 1e-8)
    return data


def beat_briner_winsorize(data: pd.DataFrame) -> pd.DataFrame:
    s_plus = (0.5 / (data - 3).max(axis=1)).clip(None, 1).clip(0, None).values[:, None]
    s_minus = (-0.5 / (data + 3).min(axis=1)).clip(None, 1).clip(0, None).values[:, None]
    values = np.where(data.values > 3, 3 * (1 - s_plus) + data.values * s_plus, 0) + \
        np.where(data.values < -3, -3 * (1 - s_minus) + data.values * s_minus, 0) + \
        np.where((data.values >= -3) & (data.values <= 3), data.values, 0)
    assert -3.51 <= values.min() <= values.max() <= 3.51
    values[np.isnan(data.values)] = np.nan
    data = pd.DataFrame(values, index=data.index, columns=data.columns)
    return data.stack().sort_index()