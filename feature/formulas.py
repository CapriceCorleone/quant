'''
Author: WangXiang
Date: 2024-04-14 02:40:08
LastEditTime: 2024-04-15 00:04:08
'''

import numpy as np
import pandas as pd
from numba import njit


def has_cupy():
    try:
        import cupy as cp
        has = True
    except ModuleNotFoundError:
        has = False
    return has


HAS_CUPY = has_cupy()
if HAS_CUPY:
    import cupy as cp


def get_mod(x):
    if HAS_CUPY:
        if isinstance(x, cp.ndarray):
            __mod__ = cp
        else:
            __mod__ = np
    else:
        __mod__ = np
    return __mod__


# %%
__daily_features__ = [
    'daily_bar',
    'weekly_bar',
    'risk_factor'
]

__minute_features__ = [
    'a_apt_corr',
    'a_entropy',
    'amihud',
    'amihud_adj',
    'apb',
    'apt_down',
    'apt_up',
    'arpp',
    'bigorder_inflow_ratio',
    'buy_sell_soa_2_vol',
    'center_down',
    'center_up',
    'climb_height',
    'cloud_away',
    'hidden_flower',
    'hotspot_switch',
    'jump',
    'long_short_game',
    'long_short_game_awp',
    'lppd',
    'maxdrawdown',
    'newhigh_count',
    'newlow_count',
    'out_crowd',
    'p_entropy',
    'p_mmd',
    'r_abs_ratio',
    'r_autocorr',
    'r_bigorder_driven',
    'r_cf',
    'r_close',
    'r_ddff_abs_median',
    'r_diff_abs_mean',
    'r_entropy',
    'r_hist',
    'r_jump',
    'r_kurt',
    'r_kurt_close',
    'r_kurt_large_v',
    'r_kurt_squared',
    'r_mad',
    'r_max',
    'r_min',
    'r_mmd',
    'r_overnight',
    'r_pf',
    'r_quartile_range',
    'r_rms',
    'r_rsj',
    'r_skew',
    'r_skew_close',
    'r_skew_large_v',
    'r_skew_squared',
    'r_sma',
    'r_variation',
    'r_vol',
    'r_vol_close',
    'r_vol_down',
    'r_vol_large_v',
    'r_vol_squared',
    'r_vol_up', 
    'tide',
    'trade_high_act',
    'trend_ratio',
    'v_center_close',
    'v_center_open',
    'v_cf',
    'v_entropy',
    'v_hhi',
    'v_hist',
    'v_k_kurt',
    'v_k_margin',
    'v_k_peak',
    'v_k_pulse',
    'v_k_shape',
    'v_k_skew',
    'v_k_std',
    'v_mad',
    'v_max_pct',
    'v_min_pct',
    'v_mmd',
    'v_pct_close',
    'v_pct_open',
    'v_peak_num',
    'v_pf',
    'v_quartile_range',
    'v_rms',
    'v_rsj',
    'v_skew',
    'v_variation',
    'v_vol',
    'v_vol_down',
    'v_vol_up',
    'vol_of_rvol',
    'vol_of_vvol',
    'vp_corr',
    'vp_corr_close',
    'vp_corr_large_v',
    'vp_k_corr',
    'vppe',
    'vr_corr',
    'vr_corr_close',
    'vr_corr_lag',
    'vr_corr_lag_close',
    'vr_corr_lag_large_v',
    'vr_corr_large_v',
    'vr_corr_lead_close',
    'vr_corr_lead_large_v'
]

__all__ = __daily_features__ + __minute_features__


# %% Daily Bar Factor
def __daily_bar(AShareEODPrices, init_date = 20050101, return_adjfactor = False):
    __valid_fields__ =  ['open', 'high', 'low', 'close', 'vwap', 'volume', 'amount']
    if return_adjfactor:
        __valid_fields__.append('adjfactor')
    field_mapping = dict(zip([field.split('_')[-1].lower() for field in AShareEODPrices.columns], AShareEODPrices.columns))
    field_mapping['vwap'] = field_mapping.pop('avgprice')
    df = AShareEODPrices.loc[str(init_date):].copy()
    for f in __valid_fields__:
        if f not in ('volume', 'amount', 'adjfactor'):
            df[field_mapping[f]] = df[field_mapping[f]] * df['S_DQ_ADJFACTOR']
        if f in ('volume', 'amount'):
            df[field_mapping[f]] = np.log(df[field_mapping[f]] + 1)
    outputs = df[[field_mapping[k] for k in __valid_fields__]].astype(np.float32).unstack()
    outputs = {field: outputs[field_mapping[field]] for field in __valid_fields__}
    return outputs


def daily_bar(AShareEODPrices, init_date = 20050101):
    return __daily_bar(AShareEODPrices, init_date, return_adjfactor = False)


def weekly_bar(AShareEODPrices, init_date = 20050101):
    __valid_fields__ = ['open', 'high',' low', 'close', 'vwap', 'volume', 'amount']
    outputs_daily = __daily_bar(AShareEODPrices, init_date = init_date - 10000, return_adjfactor = True)
    outputs = {}
    adj_factor = outputs_daily.pop('adjfactor').loc[str(init_date):]
    for field in __valid_fields__:
        if field == 'open':
            result = outputs_daily[field].shift(4)
        elif field == 'close':
            result = outputs_daily[field].copy()
        elif field == 'high':
            result = outputs_daily[field].rolling(5).max()
        elif field == 'low':
            result = outputs_daily[field].rolling(5).min()
        elif field in ('volume', 'amount'):
            result = (np.exp(outputs_daily[field]) - 1).rolling(5).sum()
        field_5d = field + '5d'
        outputs[field_5d] = result.loc[str(init_date):]
    vwap5d = np.where(outputs['amount5d'] > 0, outputs['amount5d'] / outputs['volume5d'] * adj_factor * 10, outputs['close5d'])
    outputs['vwap5d'] = pd.DataFrame(vwap5d, columns=outputs['close5d'].columns, index=outputs['close5d'].index)
    return outputs


def risk_factor(FactorExposure, init_date = 20050101):
    outputs = {k: v.loc[init_date:] for k, v in FactorExposure}
    return outputs


# %% Basic Functions
def __skew(x, axis=0):
    x_3 = np.mean(np.power(x * 100 - np.mean(x * 100, axis=axis), 3), axis=axis)
    return x_3 / np.power(np.std(x * 100, axis=axis), 3)


def __kurt(x, axis=0):
    x_4 = np.mean(np.power(x - np.mean(x, axis=axis), 4), axis=axis)
    return x_4 / np.power(np.std(x, axis=axis), 4) - 3


def __nanskew(x, axis=0):
    return __skew(np.nan_to_num(x, nan=0.0), axis=axis)


def __nankurt(x, axis=0):
    return __kurt(np.nan_to_num(x, nan=0.0), axis=axis)


def __shift(x, n):
    """ numpy矩阵移动n行位置 """
    __mod__ = get_mod(x)
    y = __mod__.zeros(x.shape) * __mod__.nan
    y = __mod__.zeros(x.shape) * __mod__.nan
    y[n:] = x[:-n]
    return y


def __diff(x, n):
    """ numpy矩阵减去前n行 """
    return (x - __shift(x, n))


def __corr_of_matrices(x, y, axis=0):
    """
    rho(a,b) = cov(a,b) / (std(a) * std(b))
    correlation between each columns(default) or rows of two matrices
    """
    n = x.shape[0] - 1
    x_ = x - x.mean(axis=axis)
    y_ = y - y.mean(axis=axis)
    covar = np.sum(x_ * y_, axis=axis) / n
    sig_x = np.sqrt(np.sum(x_ ** 2, axis=axis) / n)
    sig_y = np.sqrt(np.sum(y_ ** 2, axis=axis) / n)
    return covar / (sig_x * sig_y)


def __nancovar_of_matrices(x, y, axis=0):
    """ 求矩阵x和y的相关系数 """
    nan_valid = ~np.isnan(x) & ~np.isnan(y)
    x_, y_ = x.copy(), y.copy()
    x_[~nan_valid] = np.nan
    y_[~nan_valid] = np.nan
    n = nan_valid.sum(0) - 1
    x_ = (x_ - np.nanmean(x_, axis=axis)) / np.nanstd(x_, axis=axis)
    y_ = (y_ - np.nanmean(y_, axis=axis)) / np.nanstd(y_, axis=axis)
    covar = np.nansum(x_ * y_, axis=axis) / n
    covar[n < 0] = np.nan
    return covar


def __coefficient_of_variation(x, axis=0):
    return np.nanstd(x, axis=axis) / np.nanmean(x, axis=axis)


def __rolling(x, window):
    """
    rolling along first axis, return matrix with additional dimension of time
    """
    shape = (x.shape[0] - window + 1, window) + x.shape[1:]
    strides = (x.strides[0], ) + x.strides
    __mod__ = get_mod(x)
    return __mod__.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def __corr(x, axis=0):
    if axis == 1:
        x = x.T
    x_norm = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return np.matmul(x_norm.T, x.norm) / x.shape[0]


def __entropy(x, axis=0):
    prob = np.abs(x) / np.sum(np.abs(x), axis=axis)
    return -np.nansum(prob * np.log2(prob), axis=axis)


def __rank(x, axis=0):
    __mod__ = get_mod(x)
    sorted_indices = __mod__.argsort(x, axis=axis)
    ranks = __mod__.empty_like(sorted_indices)
    if axis == 0:
        ranks[sorted_indices, __mod__.arange(x.shape[1])] = __mod__.arange(x.shape[0])[:, None]
    elif axis == 1:
        ranks[__mod__.arange(x.shape[0])[:, None], sorted_indices] = __mod__.arange(x.shape[1])
    ranks = ranks.astype(x.dtype)
    ranks[__mod__.isnan(x)] = __mod__.nan
    return ranks


def __zscore(x, axis=0):
    return (x - np.nanmean(x, axis=axis)) / np.nanstd(x, axis=axis)


def __center_time(x):
    __mod__ = get_mod(x)
    time_ix = __mod__.arange(len(x))
    return (x.T * time_ix).sum(axis=1) / x.sum(axis=0)


def __max_min_distance(x, axis=0):
    return (np.nanargmax(x, axis=axis) - np.nanargmin(x, axis=axis)).astype(x.dtype)


def __percentile_ratio(x, percentile=0.2, order='ascend'):
    """
    Calculates the percentile ratio of a given two-dimensional array along the first axis.

    Parameters:
    - x (ndarray): Input array for calculating the percentile ratio. Should be two-dimensional.
    - percentile (float, optional): The desired percentile (default is 2.0).
    - order (str, optional): The sorting order ('ascend' or 'descend', default is 'ascend').

    Returns:
    - result (ndarray): The percentile ratio calculated based on the specified parameters.
    The shape of the result will be the sample as the second axis of the input array.

    Example:
    >>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> __percentile_ratio(x, percentile = 0.5, order = 'ascend')
    array([0.08333333, 0.13333333, 0.16666667])

    >>> __percentile_ratio(x, percentile = 0.5, order = 'ascend')
    array([0.91666667, 0.86666667, 0.83333333])
    """
    if order == 'ascend':
        kth = int(x.shape[0] * percentile)
        result = np.partition(x, kth, axis=0)[:kth].sum(axis=0) / x.sum(0)
    elif order == 'descend':
        kth = int(x.shape[0] * (1 - percentile))
        result = np.partition(x, kth, axis=0)[kth:].sum(axis=0) / x.sum(0)
    return result


# %% Minute Bar Feature
def r_vol(q):
    """日内波动率"""
    factor = np.nanstd(q.ret * 100, axis=0, ddof=1)
    return factor


def r_skew(q):
    """日内偏度"""
    rmean = np.nanmean(q.ret * 100, axis=0)
    factor = np.nansum((q.ret * 100 - rmean) ** 3, axis=0) / len(q.ret) / r_vol(q) ** 3
    return factor


def r_kurt(q):
    """日内峰度"""
    rmean = np.nanmean(q.ret * 100, axis=0)
    factor = np.nansum((q.ret * 100 - rmean) ** 4, axis=0) / q.ret.shape[0] / r_vol(q) ** 4
    return factor


def v_vol(q):
    """日内成交量波动率"""
    vlm = q.volume / np.nansum(q.volume, axis=0)
    factor = np.nanstd(vlm, axis=0, ddof=1)
    return factor


def v_skew(q):
    """日内成交量偏度"""
    vlm = q.volume / np.nansum(q.volume, axis=0)
    vmean = np.nanmean(vlm, axis=0)
    factor = np.nansum((vlm - vmean) ** 3, axis=0) / vlm.shape[0] / v_vol(q) ** 3
    return factor


def v_kurt(q):
    """日内成交量峰度"""
    vlm = q.volume / np.nansum(q.volume, axis=0)
    vmean = np.nanmean(vlm, axis=0)
    factor = np.nansum((vlm - vmean) ** 4, axis=0) / vlm.shape[0] / v_vol(q) ** 4
    return factor


def v_hhi(q):
    """日内成交量HHI指数"""
    vlm = q.volume / np.nansum(q.volume, axis=0)
    factor = np.nansum(vlm ** 2, axis=0) * 100
    return factor


def twap(q):
    """TWAP"""
    factor = np.nanmean((q.open + q.close + q.high + q.low) / 4, axis=0)
    return factor


def r_jump(q):
    """日内极端收益率"""
    rets = np.log(q.close / q.preclose_2) * 100
    rmedian = np.nanmedian(rets, axis=0)
    rmad = np.nanmedian(np.abs(rets - rmedian), axis=0) * 1.483
    is_outlier = (np.abs(rets - rmedian) > rmad * 1.96).astype(int)
    factor = np.nansum(rets * is_outlier, axis=0)
    return factor


def r_close(q):
    """尾盘收益率"""
    factor = np.nansum(q.ret[-30:], axis=0)
    return factor


def v_pct_open(q):
    """早盘成交量占比"""
    factor = np.nansum(q.volume[:30], axis=0) / np.nansum(q.volume, axis=0)
    return factor


def v_pct_close(q):
    """尾盘成交量占比"""
    factor = np.nansum(q.volume[-30:], axis=0) / np.nansum(q.volume, axis=0)
    return factor


def r_vol_squared(q):
    """日内波动率平方"""
    return r_vol(q) ** 2


def r_skew_squared(q):
    """日内偏度平方"""
    return r_skew(q) ** 2


def r_kurt_squared(q):
    """日内峰度平方"""
    return r_kurt(q) ** 2


def amihud(q):
    """Amihud非流动性"""
    amt = q.amount.copy()
    amt[amt <= 0] = np.nan
    factor = np.nansum(np.abs(q.ret) / amt, axis=0) / q.ret.shape[0]
    return factor


def amihud_adj(q):
    rets = q.ret * 100
    amt = q.amount.copy()
    amt[amt <= 0] = np.nan
    ls_illiq = np.nanmean(np.abs(rets), axis=0) * np.nanmean(1 / amt)
    cov = np.nanmean(np.abs(rets) * amt, axis=0) - np.nanmean(np.abs(rets), axis=0) * np.nanmean(amt, axis=0)
    adj = cov / np.nanmean(amt, axis=0) ** 2
    factor = ls_illiq - adj
    return factor


def r_vol_up(q):
    """上行波动率"""
    r = q.ret.copy()
    r[q.ret <= 0] = np.nan
    factor = np.nanstd(r, axis=0, ddof=1)
    return factor


def r_vol_down(q):
    """下行波动率"""
    r = q.ret.copy()
    r[q.ret >= 0] = np.nan
    factor = np.nanstd(r, axis=0, ddof=1)
    return factor


def r_rsj(q):
    """RSJ"""
    rets = q.ret * 100
    rets_positive, rets_negative = rets.copy(), rets.copy()
    rets_positive[rets <= 0] = np.nan
    rets_negative[rets >= 0] = np.nan
    factor = (np.nansum(rets_negative ** 2, axis=0) - np.nansum(rets_positive ** 2, axis=0)) / (np.nansum(rets ** 2, axis=0) + 1e-8)
    return factor


def v_rsj(q):
    """成交量RSJ"""
    rets = q.ret * 100
    vlm = q.volume
    vlm = vlm / np.nansum(vlm, axis=0)
    vlm_positive, vlm_negative = vlm.copy(), vlm.copy()
    vlm_positive[rets <= 0] = np.nan
    vlm_negative[rets >= 0] = np.nan
    factor = (np.nansum(vlm_negative ** 2, axis=0) - np.nansum(vlm_positive ** 2, axis=0)) / (np.nansum(vlm ** 2, axis=0) + 1e-8)
    return factor


def trend_ratio(q):
    """趋势占比"""
    close = q.close
    factor = (close[-1] - close[0]) / (np.nansum(np.abs(np.diff(close, axis=0)), axis=0) + 1e-8)
    return factor


def maxdrawdown(q):
    """日内最大回撤"""
    cum_high = np.tile(q.high, (q.shape[0], 1, 1))
    cum_high[np.triu_indices(240, 1)] = 0
    cum_high = cum_high.max(axis=1)
    return (q.low / cum_high - 1).min(axis=0)


def vp_corr(q):
    """价量相关性"""
    return __corr_of_matrices(q.close, q.volume, axis=0)


def vr_corr(q):
    """收益率与量的相关性"""
    return __corr_of_matrices(q.ret, q.volume, axis=0)


def vr_corr_lag(q):
    """滞后收益率与量的相关性"""
    return __corr_of_matrices(q.ret[:-1], q.volume[1:], axis=0)


def vr_corr_lead(q):
    """超前收益率与量的相关性"""
    return __corr_of_matrices(q.ret[1:], q.volume[:-1], axis=0)


def r_overnight(q):
    """隔夜收益率"""
    return q.open[0] / q.preclose[0] - 1


def vp_corr_close(q):
    """尾盘价量相关性"""
    return __corr_of_matrices(q.close[-30:], q.volume[-30:], axis=0)


def vr_corr_close(q):
    """尾盘收益率与量的相关性"""
    return __corr_of_matrices(q.ret[-30:], q.volume[-30:], axis=0)


def vr_corr_lag_close(q):
    """尾盘滞后收益率与量的相关性"""
    return __corr_of_matrices(q.ret[-30:-1], q.volume[-29:], axis=0)


def vr_corr_lead_close(q):
    """尾盘超前收益率与量的相关性"""
    return __corr_of_matrices(q.ret[-29:], q.volume[-30:-1], axis=0)


def r_vol_close(q):
    """尾盘波动率"""
    return np.nanstd(q.ret[-30:] * 100, axis=0, ddof=1)


def r_skew_close(q):
    """尾盘日内偏度"""
    rets = q.ret[-30:] * 100
    return __nanskew(rets, axis=0)


def r_kurt_close(q):
    """尾盘日内峰度"""
    rets = q.ret[-30:] * 100
    return __nankurt(rets, axis=0)


def r_vol_large_v(q):
    """大成交量对应的波动率"""
    loc = np.argpartition(q.volume, -80, axis=0)[-80:]
    factor = np.nanstd(np.take_along_axis(q.ret * 100, loc, axis=0), axis=0, ddof=1)
    return factor


def r_skew_large_v(q):
    """大成交量对应的日内偏度"""
    loc = np.argpartition(q.volume, -80, axis=0)[-80:]
    rets = np.take_along_axis(q.ret * 100, loc, axis=0)
    return __nanskew(rets, axis=0)


def r_kurt_large_v(q):
    """大成交量对应的日内峰度"""
    loc = np.argpartition(q.volume, -80, axis=0)[-80:]
    rets = np.take_along_axis(q.ret * 100, loc, axis=0)
    return __nankurt(rets, axis=0)


def r_vol_ratio_large_v(q):
    """大成交量波动率占比"""
    return r_vol_large_v(q) / r_vol(q)


def vp_corr_large_v(q):
    """大成交量对应的价量相关性"""
    loc = np.argpartition(q.volume, -80, axis=0)[-80:]
    close = np.take_along_axis(q.close, loc, axis=0)
    volume = np.take_along_axis(q.volume, loc, axis=0)
    return __corr_of_matrices(close, volume)


def vp_corr_large_v(q):
    """大成交量对应的收益率与量的相关性"""
    loc = np.argpartition(q.volume, -80, axis=0)[-80:]
    rets = np.take_along_axis(q.ret, loc, axis=0)
    volume = np.take_along_axis(q.volume, loc, axis=0)
    return __corr_of_matrices(rets, volume)


def vp_corr_lag_large_v(q):
    """大成交量对应的滞后收益率与量的相关性"""
    loc = np.argpartition(q.volume[1:], -80, axis=0)[-80:]
    rets = np.take_along_axis(q.ret, loc, axis=0)
    volume = np.take_along_axis(q.volume[1:], loc, axis=0)
    return __corr_of_matrices(rets[:-1], volume[1:])


def vp_corr_lead_large_v(q):
    """大成交量对应的超前收益率与量的相关性"""
    loc = np.argpartition(q.volume[:-1], -80, axis=0)[-80:]
    rets = np.take_along_axis(q.ret[1:], loc, axis=0)
    volume = np.take_along_axis(q.volume, loc, axis=0)
    return __corr_of_matrices(rets[1:], volume[:-1])


def vol_of_rvol(q):
    rets = q.ret * 100
    return np.nanstd(np.nanstd(rets.reshape(len(rets) // 5, 5, -1), axis=0), axis=0)


def vol_of_vvol(q):
    vlm = q.volume / np.nansum(q.volume, axis=0)
    return np.nanstd(np.nanstd(vlm.reshape(len(vlm) // 5, 5, -1), axis=0), axis=0)


def r_mad(q):
    rets = q.ret * 100
    return np.nanmedian(np.abs(rets - np.nanmedian(rets, axis=0)), axis=0)


def v_mad(q):
    vlm = q.volume / np.nansum(q.volume, axis=0)
    return np.nanmedian(np.abs(vlm - np.nanmedian(vlm, axis=0)), axis=0)


def __center_of_ret(q, ret):
    """
    center of returns, from tgd
    """
    s = np.arange(1, q.shape[0] + 1)
    if q.cupy:
        s = cp.array(s)
    return np.sum(np.sign(ret) * s[:, (None)] * ret, axis=0) / np.sum(ret, axis=0)


def center_up(q):
    return __center_of_ret(q, np.maximum(q.ret, 0))


def center_down(q):
    return __center_of_ret(q, -np.minimum(q.ret, 0))


def jump(q):
    """
    return jumps from flying_moth
    """
    ret_c = np.log(1 + q.ret)
    jumps = 2 * (q.ret - ret_c) - ret_c ** 2
    return {
        'jumps': jumps.mean(axis=0) * 1_0000_0000.0,
        'jumps_stability': jumps.std(axis=0) * 1_0000_0000.0
    }


def climb_height(q):
    """
    better vol from climb_height
    """
    price_ix = np.where(np.isin(q.fields, ['open', 'high', 'low', 'close']))[0]
    prices = np.transpose(q.data[:, :, (price_ix)], axes=(0, 2, 1))
    rolling_prices = __rolling(prices, 5).reshape(q.shape[0] - 4, 20, -1)
    bvol = (rolling_prices.std(axis=1) / rolling_prices.mean(axis=1)) ** 2
    r2bvol = q.ret[-bvol.shape[0]:] / bvol
    cov_ret_bvol = np.nanmean((r2bvol - np.nanmean(r2bvol, axis=0)) * (bvol - np.nanmean(bvol, axis=0)), axis=0) * 1_0000.0
    bvol[bvol < bvol.mean(axis=0) + bvol.std(axis=0)] = np.nan
    r2bvol = q.ret[-bvol.shape[0]:] / bvol
    cov_ret_bvol_extreme = np.nanmean((r2bvol - np.nanmean(r2bvol, axis=0)) * (bvol - np.nanmean(bvol, axis=0)), axis=0)
    return {
        'cov_ret_bvol': cov_ret_bvol,
        'cov_ret_bvol_extreme': cov_ret_bvol_extreme
    }


def tide(q):
    def find_tide_indices(volume_data, row_index, peak_index, data_shape):
        mask_rising_tide = row_index < peak_index
        mask_falling_tide = row_index > peak_index
        rising_tide_index = np.nanargmin(np.where(mask_rising_tide, volume_data, np.inf), axis=0)
        falling_tide_index = np.nanargmin(np.where(mask_falling_tide, volume_data, np.inf), axis=0)
        rising_tide_index[peak_index == 4] = 0
        falling_tide_index[peak_index == data_shape - 5] = data_shape - 1
        return rising_tide_index, falling_tide_index
    
    def calculate_tide(prices, indices, key1, key2):
        return (prices[key1] / prices[key2] - 1) / (indices[key1] - indices[key2]) * 100
    
    volume_data = q.close * np.nan
    volume_data[4:-4] = __rolling(q.volume, 9).sum(axis=1)
    peak_index = np.nanargmax(volume_data, 0)
    row_index = np.arange(q.shape[0])[:, (None)]
    if q.cupy:
        row_index = cp.array(row_index)
    rising_tide_index, falling_tide_index = find_tide_indices(volume_data, row_index, peak_index, q.shape[0])
    indices = {'rising': rising_tide_index, 'falling': falling_tide_index, 'peak': peak_index}
    volumes = {key: np.take_along_axis(volume_data, indices[key][(None), :], axis=0) for key in indices}
    prices = {key: np.take_along_axis(q.close, indices[key][(None), :], axis=0) for key in indices}
    tide_calculations = {
        'tide_full': calculate_tide(prices, indices, 'falling', 'rising'),
        'tide_rising': calculate_tide(prices, indices, 'peak', 'rising'),
        'tide_falling': calculate_tide(prices, indices, 'falling', 'peak')
    }
    for tide_type in ['tide_strong', 'tide_weak']:
        volume_condition = volumes['falling'] > volumes['rising'] if tide_type == 'tide_strong' else volumes['falling'] < volumes['rising']
        tide_value = calculate_tide(prices, indices, 'peak', 'rising')
        tide_value[volume_condition] = calculate_tide(prices, indices, 'falling', 'rising')[volume_condition]
    return tide_calculations


def cloud_away(q):
    vol = __rolling(q.ret, 5).std(axis=1)
    ambiguity = __rolling(vol, 5).std(axis=1) * 100
    amount = q.amount[-ambiguity.shape[0]:]
    volume = q.volume[-ambiguity.shape[0]:]
    ix = ambiguity > np.nanmean(ambiguity, axis=0)
    return {
        'ambiguity_amount_corr': __corr_of_matrices(ambiguity, amount, axis=0),
        'ambiguity_amount_pct': np.sum(amount * ix, axis=0) / amount.sum(axis=0),
        'ambiguity_quantity_pct': np.sum(volume * ix, axis=0) / volume.sum(axis=0)
    }


def out_crowd(q):
    """
    out_crowd from technical factor "boat_sail"
    """
    s = q.ret.std(axis=1)
    amount_corr = __corr(q.amount[s <= s.mean()])
    return (amount_corr.sum(axis=0) - 1) / (amount_corr.shape[0] - 1)


def hidden_flower(q):
    """
    Do lienar regression on multiple sample simultaneously with einstein summation
    """
    L = 6
    n = len(q)
    volume_diff = q.volume[1:] - q.volume[:-1]
    y = q.ret[L:].T
    x = __rolling(volume_diff, volume_diff.shape[0] - L + 1).transpose((2, 1, 0))
    t, f = x.shape[1], x.shape[2]
    __mod__ = get_mod(q.close)
    ones = __mod__.ones((n, t, 1))
    x = __mod__.concatenate([x, ones], axis=2)
    xTx = __mod__.einsum("ntf,ntg->nfg", x, x)
    xTx_inv = __mod__.linalg.inv(xTx)
    betas = __mod__.einsum("nfg,nft,nt->ng", xTx_inv, x.transpose((0, 2, 1)), y)
    y_preds = __mod__.einsum("nf,ntf->nt", betas, x)
    residuals = y - y_preds
    y_means = __mod__.mean(y, axis=1)
    ssrs = __mod__.sum(residuals ** 2, axis=1)
    ssregs = __mod__.sum((y_preds - y_means[:, (None)]) ** 2, axis=1)
    mses = ssrs / (t - f)
    t_stats = betas / __mod__.sqrt(__mod__.diagonal(xTx_inv, offset=0, axis1=1, axis2=2) * mses[:, (None)])
    t_std = t_stats[:, :-2].std(axis=1)
    f_stats = ssregs / (L - 1) / mses
    t_intercepts = t_stats[:, (0)]
    return {
        'v_t_stat_std': t_std,
        'v_intercept_t_stat': t_intercepts,
        'v_f_stat': f_stats
    }


def long_short_game(q):
    """
    多空博弈：收益率正道许成交量累加和做差之和
    参考文献：方正证券《股票日内多空博弈激烈程度度量与“多空博弈”因子构建》
    逻辑：利用收益率区分多空并根据成交量衡量多空博弈程度
    """
    volume_pct = q.volume / np.nansum(q.volume, axis=0)
    volume_pct[np.isnan(volume_pct)] = 0
    ind = np.arange(volume_pct.shape[1])
    rrank = np.argsort(q.ret, axis=0)
    spower = volume_pct[rrank, ind]
    lpower = volume_pct[np.flip(rrank, axis=0), ind]
    factor = (np.cumsum(spower, axis=0) - np.cumsum(lpower, axis=0)).sum(axis=0)
    factor = np.abs(factor - np.nanmean(factor)) / np.nanstd(factor)
    return factor


def long_short_game_awp(q):
    """
    多空博弈：收益率正倒序振幅累加和做差之和
    参考文献：方正证券《股票日内多空博弈激烈程度度量与“多空博弈”因子构建》
    """
    awp = (q.high - q.low) / q.close
    awp[np.isnan(awp)] = 0
    ind = np.arange(awp.shape[1])
    rrank = np.argsort(q.ret, axis=0)
    spower = awp[rrank, ind]
    lpower = awp[np.flip(rrank, axis=0), ind]
    factor = (np.cumsum(spower, axis=0) - np.cumsum(lpower, axis=0)).sum(axis=0)
    factor = np.abs(factor - np.nanmean(factor)) / np.nanstd(factor)
    return factor


def trade_high_act(q):
    """
    高活跃每笔成交：成交量前20%的部分每笔成交量占总计每笔成交量的比重
    参考文献：长江证券高频因子
    逻辑：刻画活跃成交时的大单占比
    """
    cjbs = np.nansum(q.cjbs, axis=0)
    cjbs[cjbs == 0] = np.nan
    volume_pertrans = np.nansum(q.volume, axis=0) / cjbs
    vrank = np.argsort(-q.volume, axis=0)
    ind = np.arange(q.volume.shape[1])
    vol = q.volume[vrank, ind]
    cj = q.cjbs[vrank, ind]
    cj_top20 = np.nansum(cj[:int(vrank.shape[0] / 5), :], axis=0)
    cj_top20[cj_top20 == 0] = np.nan
    factor = np.nansum(vol[:int(vrank.shape[0] / 5), :], axis=0) / cj_top20 / volume_pertrans
    return factor


def v_peak_num(q):
    """
    成交量波峰计数：成交量大于均值+一倍标准差的非连续值计数
    参考文献：长江证券《高频波动中的时间序列信息》
    逻辑：刻画趋势交易者行为
    """
    peak = np.where((q.volume - (np.nanmean(q.volume, axis=0) + np.nanstd(q.volume, axis=0)) > 0), 1, 0)
    return np.where(peak[1:, :] > peak[:-1, :], 1, 0).sum(axis=0) + peak[(0), :]


