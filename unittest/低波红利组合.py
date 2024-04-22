import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('E:/')
os.chdir('E:/quant/projects/状态波动率/')

from quant.active import ActivePortfolio
from quant.core import Aligner, DataLoader, Universe, format_unstack_table, winsorize_mad, stdd_zscore, orthogonalize_monthend
from quant.portfolio_test import PortfolioAnalyzer

aligner = Aligner()


# %% tool functions
def get_nth_value(data, n):
    result = np.zeros(len(data)) * np.nan
    for i in range(len(data)):
        value = np.sort(data.iloc[i].dropna().values)
        if len(value) > abs(n):
            result[i] = value[n]
        elif len(value) > 0:
            result[i] = value.min() if n < 0 else value.max()
    return result


def load_factor(factor_name, unstack=True, universe=None, ap=None, stan=False, orth=False):
    aligner = Aligner()
    factor = pd.read_pickle(f'./data/{factor_name}.pkl')
    if unstack:
        factor = format_unstack_table(factor.pivot(index='trade_date', columns='ticker', values='factor'))
    factor = aligner.align(factor)
    if universe is not None:
        factor[universe.astype(int) == 0] = np.nan
    if stan:
        factor = stdd_zscore(winsorize_mad(factor))
    if orth:
        factor = orthogonalize_monthend(factor, ap.risk_model['lncap'])
    return factor


# %% 中证低波红利（h20269.CSI），波动率替换
'''universe'''
# 日均成交额
df = pd.read_pickle('./data/amount_past_1y.pkl')
universe = df >= df.quantile(0.2, axis=1).values[:, None]

# 日均总市值
df = pd.read_pickle('./data/mkts_past_1y.pkl')
universe = universe & (df >= df.quantile(0.2, axis=1).values[:, None])

# 过去三年连续分红
df = [
    pd.read_pickle('./data/dividend_1y.pkl'),
    pd.read_pickle('./data/dividend_2y.pkl'),
    pd.read_pickle('./data/dividend_3y.pkl'),
]
universe = universe & (df[0] != df[1]) \
                    & (df[0] != df[2]) \
                    & (df[1] != df[2]) \
                    & (df[0].fillna(value=0) > 0) \
                    & (df[1].fillna(value=0) > 0) \
                    & (df[2].fillna(value=0) > 0)

# 股利支付率
df = pd.read_pickle('./data/pay_ratio_1y.pkl')
universe = universe & ((df >= 0) & (df <= df.quantile(0.95, axis=1).values[:, None]))

# 每股股利增长率
df = pd.read_pickle('./data/dps_cagr_3y.pkl')
# universe = universe & (df[0] >= 0) & ((df[0] - df[1]) <= 0)
# universe = universe & (2 * df[0] >= df[1])
universe = universe & (df >= 0)

# 过去三年平均税后现金股息率
df = pd.read_pickle('./data/dp_ttm_mean_3y.pkl')
df[universe.astype(int) == 0] = np.nan
universe = universe & (df >= get_nth_value(df, -75)[:, None])

'''prepare'''
dl = DataLoader()
univ = Universe()
ap = ActivePortfolio(20130101, 20240329)

'''weight'''
weight_name = 'dp_ttm'
weights = pd.read_pickle(f'./data/{weight_name}.pkl')
weights[universe.astype(int) == 0] = np.nan

'''factors (active)'''
factor_name_long = 'compose_cate1_2_cate2_2_cate1_4_cate2_5'
factor_name = '状态波动率_合成4'
factor = load_factor(factor_name_long, universe=universe, ap=ap)
factor_stan = stdd_zscore(winsorize_mad(factor))
factor_orth = orthogonalize_monthend(factor_stan, ap.risk_model['lncap'])

'''frequency'''
frequency = 'M'

'''backtest'''
result = [
    ap.calc_portfolio_daily_return({factor_name: (factor, 50)}, {'默认': universe}, {weight_name: weights}, frequency),
    ap.calc_portfolio_daily_return({f'{factor_name}_orth': (factor_orth, 50)}, {'默认': universe}, {weight_name: weights}, frequency),
]

'''figure'''
active_portfolios = pd.concat([i[0] for i in result], axis=1)
active_bmk_names = active_portfolios.columns[:1].tolist()
nav = ap.figure(active_portfolios, active_bmk_names, index_bmk_codes=['h20269.CSI'], index_bmk_names=['红利低波全收益'], grid=True, figsize=(9, 6))
perf = ap.calc_performance(nav)

'''final output'''
result = [
    ap.calc_portfolio_daily_return({f'{factor_name}_orth': (factor_orth, 50)}, {'默认': universe}, {weight_name: weights}, frequency),
]
active_portfolios = pd.concat([i[0] for i in result], axis=1)
nav = ap.figure(active_portfolios, [], index_bmk_codes=['h20269.CSI'], index_bmk_names=['红利低波全收益'],
                with_excess_names=True, grid=True, figsize=(9, 6))
perf = ap.calc_performance(nav)
perf_by_year = ap.calc_performance_by_year(nav)



# %% 红利成长低波（931130.CSI，h21130.CSI），波动率替换
'''universe'''
# 日均成交额
df = pd.read_pickle('./data/amount_past_1y.pkl')
universe = df >= df.quantile(0.2, axis=1).values[:, None]

# 日均总市值
df = pd.read_pickle('./data/mkts_past_1y.pkl')
universe = universe & (df >= df.quantile(0.2, axis=1).values[:, None])

# 过去三年连续分红
df = [
    pd.read_pickle('./data/dividend_1y.pkl'),
    pd.read_pickle('./data/dividend_2y.pkl'),
    pd.read_pickle('./data/dividend_3y.pkl'),
]
universe = universe & (df[0] != df[1]) \
                    & (df[0] != df[2]) \
                    & (df[1] != df[2]) \
                    & (df[0].fillna(value=0) != 0) \
                    & (df[1].fillna(value=0) != 0) \
                    & (df[2].fillna(value=0) != 0)

# 过去三年净利润变化
df = [
    pd.read_pickle('./data/np_ttm_cagr_3y.pkl'),
    pd.read_pickle('./data/ep_ttm_1y.pkl')      
]
universe = universe & ((df[0] >= 0) | (df[1] >= 0))

# 预期股息率
df = pd.read_pickle('./data/expected_dp.pkl')
universe = universe & (df >= df.quantile(0.4, axis=1).values[:,None])

# ROE增速波动率
df = pd.read_pickle('./data/delta_roe_std_3y.pkl')
universe = universe & (df <= df.quantile(0.6, axis=1).values[:,None])

# 过去四个季度 ROE的环比增速
df = pd.read_pickle('./data/delta_roe_1y.pkl')
universe = universe & (df >= df.quantile(0.4, axis=1).values[:,None])

# 股息率TTM（额外增加）
dp_factor_name = 'dp_ttm'
df = pd.read_pickle(f'./data/{dp_factor_name}.pkl')
df[universe.astype(int) == 0] = np.nan
dp_thresh = 150
universe_dp_selected = universe & (df >= get_nth_value(df, -dp_thresh)[:, None])

'''prepare'''
dl = DataLoader()
univ = Universe()
ap = ActivePortfolio(20130101, 20240329)

'''weight'''
weight_name = 'expected_dp'
weights = pd.read_pickle(f'./data/{weight_name}.pkl')
weights[universe.astype(int) == 0] = np.nan

'''factors (active)'''
factor_name_long = 'compose_cate1_2_cate2_2_cate1_4_cate2_5'
factor_name = '状态波动率_合成4'
factor = load_factor(factor_name_long, universe=universe, ap=ap)
factor_stan = stdd_zscore(winsorize_mad(factor))
factor_orth = orthogonalize_monthend(factor_stan, ap.risk_model['lncap'])

'''frequency'''
frequency = 'M'

'''backtest'''
result = [
    ap.calc_portfolio_daily_return({factor_name: (factor, 50)}, {'默认': universe}, {weight_name: weights}, frequency),
    ap.calc_portfolio_daily_return({factor_name: (factor, 50)}, {f'默认_{dp_factor_name}_L{dp_thresh}': universe_dp_selected}, {weight_name: weights}, frequency),
    ap.calc_portfolio_daily_return({f'{factor_name}_orth': (factor_orth, 50)}, {'默认': universe}, {weight_name: weights}, frequency),
    ap.calc_portfolio_daily_return({f'{factor_name}_orth': (factor_orth, 50)}, {f'默认_{dp_factor_name}_L{dp_thresh}': universe_dp_selected}, {weight_name: weights}, frequency),
]

'''figure'''
active_portfolios = pd.concat([i[0] for i in result], axis=1)
active_bmk_names = active_portfolios.columns[:1].tolist()
nav = ap.figure(active_portfolios, active_bmk_names, index_bmk_codes=['h21130.CSI'], index_bmk_names=['红利成长低波全收益'], grid=True, figsize=(9, 6))
perf = ap.calc_performance(nav)

'''final output'''
result = [
    ap.calc_portfolio_daily_return({f'{factor_name}_orth': (factor_orth, 50)}, {'默认': universe}, {weight_name: weights}, frequency),
]
active_portfolios = pd.concat([i[0] for i in result], axis=1)
nav = ap.figure(active_portfolios, [], index_bmk_codes=['h21130.CSI'], index_bmk_names=['红利成长低波全收益'],
                with_excess_names=True, grid=True, figsize=(9, 6))
perf = ap.calc_performance(nav)
perf_by_year = ap.calc_performance_by_year(nav)


# %% 红利成长低波（自建组合，931130.CSI，基准：h21130.CSI）
'''universe'''
# 日均成交额
df = pd.read_pickle('./data/amount_past_1y.pkl')
universe = df >= df.quantile(0.2, axis=1).values[:, None]

# 日均总市值
df = pd.read_pickle('./data/mkts_past_1y.pkl')
universe = universe & (df >= df.quantile(0.2, axis=1).values[:, None])

# 过去三年连续分红
df = [
    pd.read_pickle('./data/dividend_1y.pkl'),
    pd.read_pickle('./data/dividend_2y.pkl'),
    pd.read_pickle('./data/dividend_3y.pkl'),
]
universe = universe & (df[0] != df[1]) \
                    & (df[0] != df[2]) \
                    & (df[1] != df[2]) \
                    & (df[0].fillna(value=0) != 0) \
                    & (df[1].fillna(value=0) != 0) \
                    & (df[2].fillna(value=0) != 0)

# 过去三年净利润变化 & PE_TTM
df = [
    pd.read_pickle('./data/np_ttm_cagr_3y.pkl'),
    pd.read_pickle('./data/ep_ttm_1y.pkl')      
]
universe = universe & ((df[0] >= 0) | (df[1] >= 0))

# 股利支付率
df = pd.read_pickle('./data/pay_ratio_1y.pkl')
universe = universe & (df >= df.quantile(0.4, axis=1).clip(0, None).values[:, None])

# 预期股息率
df = pd.read_pickle('./data/expected_dp.pkl')
universe = universe & (df >= df.quantile(0.4, axis=1).values[:,None])

# ROE增速波动率
df = pd.read_pickle('./data/delta_roe_std_3y.pkl')
universe = universe & (df <= df.quantile(0.6, axis=1).values[:,None])

# 过去四个季度ROE的环比增速
df = pd.read_pickle('./data/delta_roe_1y.pkl')
universe = universe & (df >= df.quantile(0.4, axis=1).values[:,None])

# 股息率TTM（额外增加）
dp_factor_name = 'dp_ttm'
df = pd.read_pickle(f'./data/{dp_factor_name}.pkl')
df[universe.astype(int) == 0] = np.nan
dp_thresh = 150
universe_dp_selected = universe & (df >= get_nth_value(df, -dp_thresh)[:, None])

'''prepare'''
dl = DataLoader()
univ = Universe()
ap = ActivePortfolio(20130101, 20240329)

'''weight'''
weight_name = 'expected_dp'
weights = pd.read_pickle(f'./data/{weight_name}.pkl')
weights[universe.astype(int) == 0] = np.nan

'''factors (active)'''
factor_name_long = 'compose_cate1_2_cate2_2_cate1_4_cate2_5'
factor_name = '状态波动率_合成4'
factor = load_factor(factor_name_long, universe=universe)
factor_stan = stdd_zscore(winsorize_mad(factor))
factor_orth = orthogonalize_monthend(factor_stan, ap.risk_model['lncap'])

'''factors (benchmark)'''
factor_name_long_bmk = 'volatility_1y'
factor_name_bmk = '1年波动率'
factor_bmk = load_factor(factor_name_long_bmk, unstack=False)
factor_bmk_orth = orthogonalize_monthend(factor_bmk, factor_orth)

'''frequency'''
frequency = 'M'

'''output 1 (短期波动)'''
result = [
    ap.calc_portfolio_daily_return({f'{factor_name}_orth': (factor_orth, 50)}, {'红利成长低波组合': universe_dp_selected}, {weight_name: weights}, frequency),
]
active_portfolios = pd.concat([i[0] for i in result], axis=1)
nav = ap.figure(active_portfolios, [], index_bmk_codes=['h21130.CSI'], index_bmk_names=['红利成长低波全收益'],
                with_excess_names=True, grid=True, figsize=(12, 8))
perf = ap.calc_performance(nav)
perf_by_year = ap.calc_performance_by_year(nav)

'''output 2 (长短期波动结合)'''
result = [
    ap.calc_portfolio_daily_return({f'{factor_name}_orth': (factor_orth, 50), f'{factor_name_bmk}_orth_{factor_name}_orth': (factor_bmk_orth, 50)}, {'红利成长低波组合': universe_dp_selected}, {weight_name: weights}, frequency),
]
active_portfolios = pd.concat([i[0] for i in result], axis=1)
nav = ap.figure(active_portfolios, [], index_bmk_codes=['h21130.CSI'], index_bmk_names=['红利成长低波全收益'],
                with_excess_names=True, grid=True, figsize=(15, 10))
perf = ap.calc_performance(nav)
perf_by_year = ap.calc_performance_by_year(nav)
holding_num = pd.DataFrame({'持股数量': {pd.Timestamp(str(k)): len(v) for k, v in result[0][1][0].items()}})

'''attribution'''
pa = PortfolioAnalyzer(deal_price = 'preclose')
output = pa.A(holding=result[0][1][0], weight=result[0][1][1], index_code='h30269.CSI')
exp = pd.DataFrame({k: v.iloc[:, :10].loc[:20240229].dropna().mean() for k, v in output['exposure'].items()})


# %% 红利低波（自建组合，基准：h20269.CSI）
'''universe'''
# 日均成交额
df = pd.read_pickle('./data/amount_past_1y.pkl')
universe = df >= df.quantile(0.2, axis=1).values[:, None]

# 日均总市值
df = pd.read_pickle('./data/mkts_past_1y.pkl')
universe = universe & (df >= df.quantile(0.2, axis=1).values[:, None])

# 过去三年连续分红
df = [
    pd.read_pickle('./data/dividend_1y.pkl'),
    pd.read_pickle('./data/dividend_2y.pkl'),
    pd.read_pickle('./data/dividend_3y.pkl'),
]
universe = universe & (df[0] != df[1]) \
                    & (df[0] != df[2]) \
                    & (df[1] != df[2]) \
                    & (df[0].fillna(value=0) != 0) \
                    & (df[1].fillna(value=0) != 0) \
                    & (df[2].fillna(value=0) != 0)

# PE_TTM
df = pd.read_pickle('./data/ep_ttm_1y.pkl')      
universe = universe & (df >= 0)

# 股利支付率
df = pd.read_pickle('./data/pay_ratio_1y.pkl')
universe = universe & (df >= df.quantile(0.4, axis=1).clip(0, None).values[:, None])

# # 股利支付率
# df = pd.read_pickle('./data/pay_ratio_1y.pkl')
# universe = universe & ((df >= 0) & (df <= df.quantile(0.95, axis=1).values[:, None]))

# # 每股股利增长率
# df = pd.read_pickle('./data/dps_cagr_3y.pkl')
# universe = universe & (df >= 0)

# 预期股息率
df = pd.read_pickle('./data/expected_dp.pkl')
df = aligner.align(df)
universe = universe & (df >= df.quantile(0.4, axis=1).values[:, None])

# 过去3年ROE波动率
df = pd.read_pickle('./data/delta_roe_std_3y.pkl')
universe = universe & (df <= df.quantile(0.6, axis=1).values[:, None])

# 过去1年收益率
df = pd.read_pickle('./data/ret_1y.pkl')
universe = universe & (df >= df.quantile(0.1, axis=1).values[:, None])

# # 有息负债/有形资产
# df = pd.read_pickle('./data/tang_asset2int_debt.pkl')
# universe = universe & ~(df <= df.quantile(0.1, axis=1).values[:, None])

# 股息率TTM（额外增加）
dp_factor_name = 'expected_dp'
df = pd.read_pickle(f'./data/{dp_factor_name}.pkl')
df = aligner.align(df)
df[universe.astype(int) == 0] = np.nan
dp_thresh = 150
universe_dp_selected = universe & (df >= get_nth_value(df, -dp_thresh)[:, None])

'''prepare'''
dl = DataLoader()
univ = Universe()
ap = ActivePortfolio(20130101, 20240329)

'''weight'''
weight_name = 'dp_ttm'
weights = pd.read_pickle(f'./data/{weight_name}.pkl')
weights[universe.astype(int) == 0] = np.nan

'''factors (active)'''
factor_name_long = 'compose_cate1_2_cate2_2_cate1_4_cate2_5'
factor_name = '状态波动率_合成4'
factor = load_factor(factor_name_long, universe=universe, ap=ap)
factor_stan = stdd_zscore(winsorize_mad(factor))
factor_orth = orthogonalize_monthend(factor_stan, ap.risk_model['lncap'])

'''factors (benchmark)'''
factor_name_long_bmk = 'volatility_1y'
factor_name_bmk = '1年波动率'
factor_bmk = load_factor(factor_name_long_bmk, unstack=False)
factor_bmk_orth = orthogonalize_monthend(factor_bmk, factor_orth)

'''frequency'''
frequency = 'M'

'''output 1 (TOP50)'''
result = [
    ap.calc_portfolio_daily_return({f'{factor_name}_orth': (factor_orth, 50)}, {'红利低波组合': universe_dp_selected}, {weight_name: weights}, frequency),
]
active_portfolios = pd.concat([i[0] for i in result], axis=1)
nav = ap.figure(active_portfolios, [], index_bmk_codes=['h20269.CSI'], index_bmk_names=['红利低波全收益'],
                with_excess_names=True, grid=True, figsize=(12, 8))
perf = ap.calc_performance(nav)
perf_by_year = ap.calc_performance_by_year(nav)

'''output 2 (灵活数量)'''
# result = [
#     ap.calc_portfolio_daily_return({f'{factor_name}_orth': (factor_orth, 50), f'{factor_name_bmk}_orth_{factor_name}_orth': (factor_bmk_orth, 50)}, {'红利低波组合': universe_dp_selected}, {weight_name: weights}, frequency),
# ]
# active_portfolios = pd.concat([i[0] for i in result], axis=1)
# nav = ap.figure(active_portfolios, [], index_bmk_codes=['h20269.CSI'], index_bmk_names=['红利低波全收益'],
#                 with_excess_names=True, grid=True, figsize=(15, 10))
# perf = ap.calc_performance(nav)
# perf_by_year = ap.calc_performance_by_year(nav)