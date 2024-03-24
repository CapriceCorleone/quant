'''
Author: WangXiang
Date: 2024-03-24 18:45:17
LastEditTime: 2024-03-25 00:12:48
'''

import numpy as np
from numba import njit


@njit
def shift_period(period, n):
    # get date of n periods before
    if n == 4:
        return period - 10000
    mds = [331, 630, 930, 1231]
    md = period % 10000
    if md not in mds:
        return 0
    ix_period = mds.index(md)
    md_shift = mds[(ix_period - n) % 4]
    year_shift = period // 10000 - (3 - ix_period + n) // 4
    return year_shift * 10000 + md_shift


@njit
def get_target_period(m, period):
    if period in m[:, 1]:
        m_period = m[m[:, 1] == period]
        return m_period[m_period[:, 0] == m_period[:, 0].max()][-1]
    return None


@njit
def get_n_period_within(m, n):
    # get data of most recent n period
    m_within = m[m[:, 1] > shift_period(m[:, 1].max(), n)]
    n_period = np.unique(m_within[:, 1])
    n_period_data = np.zeros((len(n_period), m.shape[1]), dtype=np.float64)
    for i in range(len(n_period)):
        n_period_data[i] = get_target_period(m_within, n_period[i])
    return n_period_data[n_period_data[:, 1] > 0]


@njit
def ffunc_last(m):
    ann_dt = np.unique(m[:, 0])
    result = np.zeros((len(ann_dt), m.shape[1]), dtype=np.float64)
    for i in range(len(ann_dt)):
        m_within = m[m[:, 0] <= ann_dt[i]]
        result[i] = get_target_period(m_within, m_within[:, 1].max())
    return result[result[:, 1] > 0]


@njit
def ffunc_mrq(m):
    result = np.zeros((len(m), 3), dtype=np.float64)
    for i in range(len(m)):
        ann_dt, period, f = m[i]
        if m[i, 1] % 10000 != 331:
            m_hist = m[(m[:, 0] <= ann_dt) & (m[:, 1] <= period)]
            f_last_quarter = get_target_period(m_hist, shift_period(period, 1))
            if f_last_quarter is None:
                continue
            f = f - f_last_quarter[-1]
        result[i] = np.array([ann_dt, period, f])
    return result[result[:, 1] > 0]


@njit
def ffunc_ttm(m):
    result = np.zeros((len(m), 3), dtype=np.float64)
    for i in range(len(m)):
        ann_dt, period, f = m[i]
        if m[i, 1] % 10000 != 1231:
            m_hist = m[(m[:, 0] <= ann_dt) & (m[:, 1] <= period)]
            f_last_year = get_target_period(m_hist, int((period // 10000 - 1) * 10000 + 1231))
            f_last_year_quarter = get_target_period(m_hist, shift_period(period, 4))
            if f_last_year is None or f_last_year_quarter is None:
                continue
            f = f + (f_last_year[-1] - f_last_year_quarter[-1])
        result[i] = np.array([ann_dt, period, f])
    return result[result[:, 1] > 0]


@njit
def ffunc_yoy(m, method='ratio'):
    result = np.zeros((len(m), 3), dtype=np.float64)
    for i in range(len(m)):
        ann_dt, period, f = m[i]
        m_hist = m[(m[:, 0] <= ann_dt) & (m[:, 1] <= period)]
        if len(m_hist) > 0:
            f_last_year_quarter = get_target_period(m_hist, shift_period(period, 4))
            if f_last_year_quarter is not None:
                f0 = f_last_year_quarter[-1]
                if f0 != 0:
                    if method == 'ratio':
                        r = (f - f0) / abs(f0)
                    elif method == 'diff':
                        r = f - f0
                    else:
                        raise ValueError (f"Invalid method: {method}")
                    result[i] = np.array([ann_dt, period, r])
    return result[result[:, 0] > 0]


@njit
def ffunc_qoq(m, method='ratio'):
    result = np.zeros((len(m), 3), dtype=np.float64)
    for i in range(len(m)):
        ann_dt, period, f = m[i]
        m_hist = m[(m[:, 0] <= ann_dt) & (m[:, 1] <= period)]
        if len(m_hist) > 0:
            f_last_year_quarter = get_target_period(m_hist, shift_period(period, 1))
            if f_last_year_quarter is not None:
                f0 = f_last_year_quarter[-1]
                if f0 != 0:
                    if method == 'ratio':
                        r = (f - f0) / abs(f0)
                    elif method == 'diff':
                        r = f - f0
                    else:
                        raise ValueError (f"Invalid method: {method}")
                    result[i] = np.array([ann_dt, period, r])
    return result[result[:, 0] > 0]


@njit
def ffunc_sum(m, window=4):
    result = np.zeros((len(m), 3), dtype=np.float64)
    for i in range(len(m)):
        ann_dt, period, f = m[i]
        m_hist = m[(m[:, 0] <= ann_dt) & (m[:, 1] <= period)]
        m_within = get_n_period_within(m_hist, window)
        if len(m_within) > 0:
            sums = np.sum(m_within[:, 2])
            result[i] = np.array([ann_dt, period, sums])
    return result[result[:, 0] > 0]


@njit
def ffunc_mean(m, window=4):
    result = np.zeros((len(m), 3), dtype=np.float64)
    for i in range(len(m)):
        ann_dt, period, f = m[i]
        m_hist = m[(m[:, 0] <= ann_dt) & (m[:, 1] <= period)]
        m_within = get_n_period_within(m_hist, window)
        if len(m_within) > 2:
            mu = np.mean(m_within[:, 2])
            result[i] = np.array([ann_dt, period, mu])
    return result[result[:, 0] > 0]


@njit
def get_years_between(t0, t1):
    mds = [331, 630, 930, 1231]
    years = t1 // 10000 - t0 // 10000
    month0, month1 = mds.index(t0 % 100000), mds.index(t1 % 100000)
    month_years = (month1 - month0) * 3 / 12
    return years + month_years


@njit
def ffunc_cagr(m, window=12):
    result = np.zeros((len(m), 3), dtype=np.float64)
    for i in range(len(m)):
        ann_dt, period, f = m[i]
        m_hist = m[(m[:, 0] <= ann_dt) & (m[:, 1] <= period)]
        m_within = get_n_period_within(m_hist, window)
        m_within = m_within[m_within[:, 2] != 0, :]
        if len(m_within) >= 2:
            years = get_years_between(m_within[0, 1], m_within[-1, 1])
            cagr = (m_within[-1, 2] / m_within[0, 2]) ** (1 / years) - 1
            result[i] = np.array([ann_dt, period, cagr])
    return result[result[:, 0] > 0]


@njit
def ffunc_std(m, window=4):
    result = np.zeros((len(m), 3), dtype=np.float64)
    for i in range(len(m)):
        ann_dt, period, f = m[i]
        m_hist = m[(m[:, 0] <= ann_dt) & (m[:, 1] <= period)]
        m_within = get_n_period_within(m_hist, window)
        if len(m_within) > 2:
            std = np.std(m_within[:, 2])
            result[i] = np.array([ann_dt, period, std])
    return result[result[:, 0] > 0]


@njit
def ffunc_divide(x, y, non_neg=True):
    """
    return x / y according to the date and period of x
    """
    result = np.zeros((len(x), 3), dtype=np.float64)
    for i in range(len(x)):
        ann_dt, period, numerator = x[i]
        y_hist = y[(y[:, 0] <= ann_dt) & (y[:, 1] <= period)]
        if len(y_hist) > 0:
            period_y = period
            if period_y not in y_hist[:, 1]:
                period_y = y_hist[:, 1].max()
            denominator = get_target_period(y_hist, period_y)[-1]
            if non_neg:
                denominator = abs(denominator)
            if denominator != 0:
                result[i] = np.array([ann_dt, period, numerator / denominator])
    return result[result[:, 0] > 0]


@njit
def ffunc_zscore(m, window=4):
    """
    volatility adjusted score
    """
    result = np.zeros((len(m), 3), dtype=np.float64)
    for i in range(len(m)):
        ann_dt, period, f = m[i]
        m_hist = m[(m[:, 0] <= ann_dt) & (m[:, 1] <= period)]
        m_within = get_n_period_within(m_hist, window)
        if len(m_within) == window:
            mu = np.mean(m_within[:, -1])
            sigma = np.std(m_within[:, -1])
            result[i] = np.array([ann_dt, period, (f - mu) / sigma])
    return result[result[:, 0] > 0]


@njit
def ffunc_mscore(m, window=4):
    """
    volatility adjusted mean
    """
    result = np.zeros((len(m), 3), dtype=np.float64)
    for i in range(len(m)):
        ann_dt, period, f = m[i]
        m_hist = m[(m[:, 0] <= ann_dt) & (m[:, 1] <= period)]
        m_within = get_n_period_within(m_hist, window)
        if len(m_within) > 2:
            mu = np.mean(m_within[:, -1])
            sigma = np.std(m_within[:, -1])
            result[i] = np.array([ann_dt, period, mu / sigma])
    return result[result[:, 0] > 0]