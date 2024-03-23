'''
Author: WangXiang
Date: 2024-03-23 21:28:59
LastEditTime: 2024-03-24 01:34:36
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
    mean = (data * weight).sum(axis=1) / weight.sum(axis=1)
    std = data.std(axis=1)
    data = (data - mean.values[:, None]) / (std.values[:, None] + 1e-8)
    return data