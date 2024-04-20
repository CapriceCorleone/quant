'''
Author: WangXiang
Date: 2024-03-23 21:28:39
LastEditTime: 2024-04-20 16:08:38
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from numba import njit
from scipy import stats
from datetime import datetime
from bisect import bisect_left, bisect_right

from ..core import Aligner, Calendar, Universe, Processors, format_unstack_table
from ..core.njit.financial import ffunc_last, ffunc_ttm, ffunc_mean, ffunc_yoy, ffunc_divide, ffunc_cagr
from ..core.njit.analyst import afunc_coverage
from .tools import winsorize_box, weighted_stdd_zscore


def unstack_market_cap(AShareEODDerivativeIndicator: pd.DataFrame, init_date: int) -> pd.DataFrame:
    df_market_cap = AShareEODDerivativeIndicator.loc[str(init_date):, 'S_VAL_MV'].unstack() * 1.0e4
    df_market_cap.index = df_market_cap.index.astype(int)
    return df_market_cap


def exponential_weight(seq_len: int, half_life: int, dtype: np.dtype, ratio: float = 0.5):
    # starts with 1, e.g., [1, 1/2, 1/4, 1/8, ...]
    # output: 1-d array
    ratio = np.power(ratio, 1 / half_life) if half_life is None else 1
    weights = np.array([np.power(ratio, i) for i in range(seq_len)], dtype=dtype)
    return weights


@njit
def __time_series_regress(ydata, xdata, window, min_periods, weight_decay, universe):
    output_beta = np.zeros(ydata.shape) * np.nan
    output_se2 = np.zeros(ydata.shape) * np.nan
    for i in range(len(ydata)):
        if i < window - 1:
            continue
        yd = ydata[i - window + 1 : i + 1]
        xd = xdata[i - window + 1 : i + 1]
        xd_isnan = np.isnan(xd)
        for j in range(ydata.shape[1]):
            if universe[i, j] == 0:
                continue
            y = yd[:, j]
            mask = np.isnan(y) | xd_isnan
            if window - mask.sum() < min_periods:
                continue
            y = y[~mask]
            x = xd[~mask]
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
def __multivariate_time_series_regress(ydata, xdata, window, min_periods, weight_decay, universe):
    # assume that xdata has no NaN
    output_resvol = np.zeros(ydata.shape) * np.nan
    for i in range(len(ydata)):
        if i < window - 1:
            continue
        yd = ydata[i - window + 1 : i + 1]
        xd = xdata[i - window + 1 : i + 1]
        for j in range(ydata.shape[1]):
            if universe[i, j] == 0:
                continue
            y = yd[:, j]
            mask = np.isnan(y)
            if window - mask.sum() < min_periods:
                continue
            y = y[~mask]
            x = xd[~mask]
            x_ = np.append(x, np.ones((len(y), 1), dtype=x.dtype), axis=1)
            x_T = x_.T
            w = np.diag(weight_decay[~mask])
            resid = y - x_ @ np.linalg.inv(x_T @ w @ x_) @ x_T @ w @ y
            output_resvol[i, j] = resid.std()
    return output_resvol


def __rolling(x, window):
    shape = (x.shape[0] - window + 1, window) + x.shape[1:]
    strides = (x.strides[0], ) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def __rolling_nlargest_mean(x, window, num):
    strides = __rolling(x, window)
    result = []
    for i in range(strides.shape[0]):
        mask = np.argpartition(strides[i], -num, axis=0)[-num:]
        mean = np.take_along_axis(strides[i], mask, axis=0).mean(axis=0)
        result.append(mean)
    return np.stack(result)


def __rolling_nsmallest_mean(x, window, num):
    strides = __rolling(x, window)
    result = []
    for i in range(strides.shape[0]):
        mask = np.argpartition(strides[i], num, axis=0)[:num]
        mean = np.take_along_axis(strides[i], mask, axis=0).mean(axis=0)
        result.append(mean)
    return np.stack(result)


@njit
def __cross_sectional_regress(ydata, xdata, weight):
    residual = np.zeros(ydata.shape) * np.nan
    for i in range(len(ydata)):
        y = ydata[i]
        x = xdata[i]
        w = weight[i]
        mask = np.isnan(y) | np.isnan(x) | np.isnan(w)
        if len(y) - mask.sum() <= 2:
            continue
        yy = y[~mask]
        xx = np.append(x[~mask][:, None], np.ones((len(yy), 1), dtype=x.dtype), axis=1)
        xx_T = xx.T
        ww = np.diag(w[~mask])
        beta = np.linalg.inv(xx_T @ ww @ xx) @ xx_T @ ww @ yy
        residual[i] = y - beta[1] - beta[0] * x
    return residual


# %% Size
def size(AShareEODDerivativeIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    factor = AShareEODDerivativeIndicator.loc[str(init_date):, 'S_VAL_MV'].unstack()
    factor = np.log(factor + 1)
    return factor.loc[str(init_date):]


# %% Beta
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


def beta(AShareEODPrices: pd.DataFrame, AIndexEODPrices: pd.DataFrame, AShareIndustriesClassCITICS: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
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
    output_beta, output_se2 = __time_series_regress(stock_return, index_return, window, min_periods, weight_decay, universe.loc[start_date:].values)
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


# %% Trend
def trend(AShareEODPrices: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    window = kwargs['window']
    min_periods = kwargs['min_periods']
    stock_quote = AShareEODPrices.copy()
    stock_quote.loc[stock_quote['S_DQ_TRADESTATUSCODE'] == 0, 'S_DQ_PCTCHANGE'] = np.nan
    adjclose = (stock_quote['S_DQ_CLOSE'] * stock_quote['S_DQ_ADJFACTOR']).unstack()
    factor = adjclose.ewm(halflife=20, min_periods=10).mean() / adjclose.ewm(halflife=window, min_periods=min_periods).mean()
    return factor.loc[str(init_date):]


# %% Liquidity
def daily_turnover(AShareEODPrices: pd.DataFrame, AShareEODDerivativeIndicator: pd.DataFrame) -> pd.DataFrame:
    stock_quote = AShareEODPrices.copy()
    stock_quote.loc[stock_quote['S_DQ_TRADESTATUSCODE'] == 0, 'S_DQ_PCTCHANGE'] = np.nan
    amount = stock_quote['S_DQ_AMOUNT'].unstack() / 10
    float_mv = AShareEODDerivativeIndicator['S_DQ_MV'].unstack().reindex_like(amount).shift(1)
    float_mv = float_mv.where(float_mv > 0, 0)
    factor = np.log(amount / float_mv)
    factor[~np.isfinite(factor)] = np.nan
    return factor


def turnover(AShareEODPrices: pd.DataFrame, AShareEODDerivativeIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    window = kwargs['window']
    min_periods = kwargs['min_periods']
    factor = daily_turnover(AShareEODPrices, AShareEODDerivativeIndicator).rolling(window, min_periods=min_periods).mean()
    return factor.loc[str(init_date):]


def liquidity_beta(AShareEODPrices: pd.DataFrame, AShareEODDerivativeIndicator: pd.DataFrame, AIndexValuation: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    window = kwargs['window']
    min_periods = kwargs['min_periods']
    half_life = kwargs['half_life']
    ratio = kwargs['ratio']
    universe = kwargs['universe']
    aligner = Aligner()
    calendar = Calendar()
    
    start_date = calendar.get_prev_trade_date(init_date, window - 1)
    stock_turnover = aligner.align(format_unstack_table(daily_turnover(AShareEODPrices, AShareEODDerivativeIndicator))).loc[start_date:]
    index = stock_turnover.index
    columns = stock_turnover.columns
    stock_turnover = stock_turnover.values.astype(np.float32)
    index_turnover = np.log(AIndexValuation.loc(axis=0)[:, '000985.CSI']['TURNOVER'] / 100).droplevel(1)
    index_turnover.index = index_turnover.index.astype(int)
    index_turnover = index_turnover.reindex(index=aligner.trade_dates).loc[start_date:].values.astype(np.float32)
    weight_decay = exponential_weight(window, half_life, stock_turnover.dtype, ratio)
    output_beta, output_se2 = __time_series_regress(stock_turnover, index_turnover, window, min_periods, weight_decay, universe.loc[start_date:].values)
    factor = aligner.align(pd.DataFrame(output_beta, index=index, columns=columns))
    return factor.loc[init_date:]


# %% Volatility
def stdvol(AShareEODPrices: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    window = kwargs['window']
    min_periods = kwargs['min_periods']
    stock_quote = AShareEODPrices.copy()
    stock_quote.loc[stock_quote['S_DQ_TRADESTATUSCODE'] == 0, 'S_DQ_PCTCHANGE'] = np.nan
    pctchg = AShareEODPrices['S_DQ_PCTCHANGE'].unstack()
    factor = pctchg.rolling(window, min_periods=min_periods).std()
    return factor.loc[str(init_date):]


def price_range(AShareEODPrices: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    window = kwargs['window']
    min_periods = kwargs['min_periods']
    adjhigh = (AShareEODPrices['S_DQ_HIGH'] * AShareEODPrices['S_DQ_ADJFACTOR']).unstack()
    adjlow = (AShareEODPrices['S_DQ_LOW'] * AShareEODPrices['S_DQ_ADJFACTOR']).unstack()
    factor = adjhigh.rolling(window, min_periods=min_periods).max() / adjlow.rolling(window, min_periods=min_periods).min() - 1
    return factor.loc[str(init_date):]


def max_ret(AShareEODPrices: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    window = kwargs['window']
    num = kwargs['num']
    aligner = Aligner()
    calendar = Calendar()
    
    start_date = calendar.get_prev_trade_date(init_date, window - 1)
    stock_return = aligner.align(format_unstack_table(AShareEODPrices['S_DQ_PCTCHANGE'].unstack() / 100)).loc[start_date:]
    factor = __rolling_nlargest_mean(stock_return.values, window, num)
    factor = pd.DataFrame(factor, index=stock_return.index[window - 1:], columns=stock_return.columns)
    factor = aligner.align(factor)
    return factor.loc[init_date:]


def min_ret(AShareEODPrices: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    window = kwargs['window']
    num = kwargs['num']
    aligner = Aligner()
    calendar = Calendar()
    
    start_date = calendar.get_prev_trade_date(init_date, window - 1)
    stock_return = aligner.align(format_unstack_table(AShareEODPrices['S_DQ_PCTCHANGE'].unstack() / 100)).loc[start_date:]
    factor = __rolling_nsmallest_mean(stock_return.values, window, num)
    factor = pd.DataFrame(factor, index=stock_return.index[window - 1:], columns=stock_return.columns)
    factor = aligner.align(factor)
    return factor.loc[init_date:]


def ivff(AShareEODPrices: pd.DataFrame, FamaFrench3Factor: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    window = kwargs['window']
    min_periods = kwargs['min_periods']
    half_life = kwargs['half_life']
    ratio = kwargs['ratio']
    universe = kwargs['universe']
    aligner = Aligner()
    calendar = Calendar()

    start_date = calendar.get_prev_trade_date(init_date, window - 1)
    stock_quote = AShareEODPrices.copy()
    stock_quote.loc[stock_quote['S_DQ_TRADESTATUSCODE'] == 0, 'S_DQ_PCTCHANGE'] = np.nan
    stock_return = aligner.align(format_unstack_table(stock_quote['S_DQ_PCTCHANGE'].unstack())).loc[start_date:]
    index = stock_return.index
    columns = stock_return.columns
    stock_return = stock_return.values.astype(np.float32)
    ff_factor = FamaFrench3Factor.set_index(['trade_date']).reindex(index=aligner.trade_dates).loc[start_date:].fillna(value=0).values.astype(np.float32)
    weight_decay = exponential_weight(window, half_life, stock_return.dtype, ratio)
    output_resvol = __multivariate_time_series_regress(stock_return, ff_factor, window, min_periods, weight_decay, universe.loc[start_date:].values)
    factor = aligner.align(pd.DataFrame(output_resvol, index=index, columns=columns))
    return factor.loc[init_date:]


# %% Value
def ep_ttm(AShareEODDerivativeIndicator: pd.DataFrame, AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_ttm(x))
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareIncome], ['NET_PROFIT_EXCL_MIN_INT_INC'], operator, init_date)
    return factor.pivot(index='trade_date', columns='ticker', values='factor').loc[init_date:]


def bp(AShareEODDerivativeIndicator: pd.DataFrame, AShareBalanceSheet: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(x)
    df_market_cap = unstack_market_cap(AShareEODDerivativeIndicator, init_date)
    factor = Processors.value.process(df_market_cap, [AShareBalanceSheet], ['TOT_SHRHLDR_EQY_EXCL_MIN_INT'], operator, init_date)
    return factor.pivot(index='trade_date', columns='ticker', values='factor').loc[init_date:]


# %% Growth
def delta_roe(AShareIncome: pd.DataFrame, AShareBalanceSheet: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x, y):
        return ffunc_last(ffunc_mean(ffunc_yoy(ffunc_divide(ffunc_ttm(x), y), method='diff'), 12))
    factor = Processors.fundamental.process([AShareIncome, AShareBalanceSheet], ['NET_PROFIT_EXCL_MIN_INT_INC', 'TOT_SHRHLDR_EQY_EXCL_MIN_INT'], operator, init_date)
    return factor.pivot(index='trade_date', columns='ticker', values='factor').loc[init_date:]


def sales_growth(AShareIncome: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_cagr(ffunc_ttm(x), 12))
    factor = Processors.fundamental.process([AShareIncome], ['OPER_REV'], operator, init_date)
    return factor.pivot(index='trade_date', columns='ticker', values='factor').loc[init_date:]


def na_growth(AShareBalanceSheet: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    def operator(x):
        return ffunc_last(ffunc_cagr(x, 12))
    factor = Processors.fundamental.process([AShareBalanceSheet], ['TOT_SHRHLDR_EQY_EXCL_MIN_INT'], operator, init_date)
    return factor.pivot(index='trade_date', columns='ticker', values='factor').loc[init_date:]


# %% Non-Linear Size
def nls(AShareEODDerivativeIndicator: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    weight = kwargs['weight']
    aligner = Aligner()
    factor_size = size(AShareEODDerivativeIndicator, init_date)
    factor_size = aligner.align(format_unstack_table(factor_size))
    factor_size = winsorize_box(factor_size)
    factor_size = weighted_stdd_zscore(factor_size, weight).astype(weight.values.dtype)
    factor_size_cubed = factor_size ** 3
    factor = __cross_sectional_regress(factor_size_cubed.values, factor_size.values, np.sqrt(weight.values))
    factor = pd.DataFrame(factor, index=factor_size.index, columns=factor_size.columns)
    factor = aligner.align(factor)
    return factor.loc[init_date:]


# %% Certainty
def instholder_pct(AShareEODPrices: pd.DataFrame, AShareEODDerivativeIndicator: pd.DataFrame, ChinaMutualFundStockPortfolio: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    aligner = Aligner()
    calendar = Calendar()
    port_data = ChinaMutualFundStockPortfolio[ChinaMutualFundStockPortfolio['F_PRT_ENDDATE'] >= str(init_date - 20000)]
    port_data = port_data[port_data['F_PRT_ENDDATE'].str[-4:].isin(['0630', '1231'])]
    port_data = port_data[port_data['S_INFO_STOCKWINDCODE'].str[-2:].isin(['SH', 'SZ', 'BJ'])]
    port_data['UNI_ANN_DT'] = port_data['F_PRT_ENDDATE'].apply(lambda x: (int(x[:4]) + 1) * 10000 + 430 if x[-4:] == '1231' else int(x[:4]) * 10000 + 831)
    port_data = port_data[port_data['UNI_ANN_DT'] >= port_data['ANN_DATE'].astype(int)]
    port_data = port_data[['S_INFO_WINDCODE', 'F_PRT_ENDDATE', 'S_INFO_STOCKWINDCODE', 'F_PRT_STKQUANTITY', 'F_PRT_STKVALUETONAV', 'UNI_ANN_DT']]
    ann_dts = np.sort(np.unique(port_data['UNI_ANN_DT'].values))
    current_date = int(datetime.now().date().strftime('%Y%m%d'))
    ann_dts = ann_dts[ann_dts <= current_date]  # 需要考虑公布日是月底，但非交易日的情况

    trade_ann_dts = np.array([i if calendar.is_trade_date(i) else calendar.get_prev_trade_date(i, 1) for i in ann_dts])
    ann_dts_map = dict(zip(ann_dts, trade_ann_dts))
    port_data['UNI_ANN_DT'] = port_data['UNI_ANN_DT'].map(ann_dts_map)
    port_data = port_data.dropna(subset=['UNI_ANN_DT', 'S_INFO_STOCKWINDCODE', 'F_PRT_STKQUANTITY'])
    port_data['UNI_ANN_DT'] = port_data['UNI_ANN_DT'].astype(int)
    
    quantity = port_data.groupby(['UNI_ANN_DT', 'S_INFO_STOCKWINDCODE'])['F_PRT_STKQUANTITY'].sum().unstack()
    shares = AShareEODDerivativeIndicator.loc[str(init_date - 20000):, 'FLOAT_A_SHR_TODAY'].unstack()
    quantity = format_unstack_table(quantity)
    shares = format_unstack_table(shares)
    shares = shares.reindex_like(quantity)
    quantity[shares.isna() | (shares == 0)] = np.nan
    
    factor = quantity / shares / 10000
    factor = aligner.align(factor)
    is_all_nan = (factor.isna().sum(axis=1) == factor.shape[1]).values
    for i in range(factor.shape[0]):
        if is_all_nan[i]:
            factor.iloc[i] = factor.iloc[i - 1]
    close = aligner.align(format_unstack_table(AShareEODPrices['S_DQ_CLOSE'].unstack()))
    factor[pd.isna(close)] = np.nan
    return factor.loc[init_date:]


def coverage(AShareEODDerivativeIndicator: pd.DataFrame, AShareEarningEst: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    weight = kwargs['weight']
    aligner = Aligner()
    factor = Processors.analyst.process(AShareEarningEst, 'EST_NET_PROFIT', afunc_coverage, init_date)
    factor = aligner.align(format_unstack_table(factor.pivot(index='trade_date', columns='ticker', values='factor')))
    factor_size = size(AShareEODDerivativeIndicator, init_date)
    factor_size = aligner.align(format_unstack_table(factor_size))
    factor_size = winsorize_box(factor_size)
    factor_size = weighted_stdd_zscore(factor_size, weight).astype(weight.values.dtype)
    factor = __cross_sectional_regress(factor.values.astype(np.float64), factor_size.values.astype(np.float64), np.ones(factor.shape, dtype=np.float64))
    factor = pd.DataFrame(factor, index=factor_size.index, columns=factor_size.columns)
    factor = aligner.align(factor).fillna(value=0)
    return factor.loc[init_date:]


def listed_days(AShareDescription: pd.DataFrame, init_date: int, **kwargs) -> pd.DataFrame:
    aligner = Aligner()
    df = AShareDescription[['S_INFO_WINDCODE', 'S_INFO_LISTDATE', 'S_INFO_DELISTDATE']]
    df.columns = ['ticker', 'entry_date', 'remove_date']
    df['remove_date'] = df['remove_date'].fillna(value=aligner.trade_dates.max())
    df[['entry_date', 'remove_date']] = df[['entry_date', 'remove_date']].astype(int)
    tickers = np.sort(np.unique(df['ticker']))
    tickers_ix = dict(zip(tickers, np.arange(len(tickers))))
    data = np.zeros((len(aligner.trade_dates), len(tickers)))
    for i in range(len(df)):
        ticker, entry_date, remove_date = df.iloc[i]
        start = bisect_left(aligner.trade_dates, entry_date)
        end = bisect_right(aligner.trade_dates, remove_date)
        data[slice(start, end), tickers_ix[ticker]] = (pd.to_datetime(aligner.trade_dates[start:end].astype(str)) - pd.Timestamp(str(entry_date))).days.values
    factor = pd.DataFrame(data, index=aligner.trade_dates, columns=tickers)
    factor = aligner.align(format_unstack_table(factor))
    return factor.loc[init_date:]


# %% SOE
def soe(AShareEODPrices: pd.DataFrame, AShareCapitalization: pd.DataFrame, init_date, **kwargs) -> pd.DataFrame:
    aligner = Aligner()
    calendar = Calendar()
    data = AShareCapitalization[['ANN_DT', 'S_INFO_WINDCODE', 'S_SHARE_RTD_STATE', 'S_SHARE_RTD_STATEJUR', 'S_SHARE_NTRD_STATE', 'S_SHARE_NTRD_STATJUR', 'TOT_SHR', 'CHANGE_DT', 'CHANGE_DT1']]
    data = data[data['ANN_DT'] >= str(init_date - 10000)]
    data['ratio'] = (data['S_SHARE_RTD_STATE'] + data['S_SHARE_RTD_STATEJUR'] + data['S_SHARE_NTRD_STATE'] + data['S_SHARE_NTRD_STATJUR']) / data['TOT_SHR']
    data = data.sort_values(['S_INFO_WINDCODE', 'ANN_DT', 'CHANGE_DT', 'CHANGE_DT1']).drop_duplicates(['S_INFO_WINDCODE', 'ANN_DT'], keep='last')
    data = data.set_index(['ANN_DT', 'S_INFO_WINDCODE'])['ratio'].sort_index().unstack()
    data = format_unstack_table(data)
    day_map = {day: day if calendar.is_trade_date(day) else calendar.get_prev_trade_date(day, 1) for day in data.index}
    data = data.stack().reset_index()
    data.columns = ['ANN_DT', 'ticker', 'ratio']
    data['trade_date'] = data['ANN_DT'].map(day_map)
    data = data.sort_values(['ticker', 'trade_date', 'ANN_DT']).drop_duplicates(['trade_date', 'ticker'], keep='last')
    data = format_unstack_table(data.pivot(index='trade_date', columns='ticker', values='ratio'))
    factor = aligner.align(data).fillna(method='ffill', limit=243).fillna(value=0)
    close = aligner.align(format_unstack_table(AShareEODPrices['S_DQ_CLOSE'].unstack()))
    factor[pd.isna(close)] = np.nan
    return factor.loc[init_date:]