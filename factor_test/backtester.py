'''
Author: WangXiang
Date: 2024-03-20 22:36:50
LastEditTime: 2024-03-27 19:18:41
'''

import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from ..core import DataLoader, Universe, Calendar, Aligner

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


class FactorTester:

    ANNUALIZE_MULTIPLIER = {
        'D': 252,
        'W': 52,
        'M': 12,
        'Q': 4,
        'Y': 1,
        5:   52,
        10:  26,
        20:  12
    }

    # RISK_STYLE_FACTORS = ['beta', 'earnings_yield', 'growth', 'leverage', 'liquidity', 'momentum', 'nlsize', 'size', 'value', 'volatility']
    RISK_STYLE_FACTORS = ['size', 'beta', 'trend', 'liquidity', 'volatility', 'value', 'growth', 'nls', 'certainty', 'soe']

    RISK_INDUSTRY_FACTORS = [
        '交通运输', '传媒', '农林牧渔', '医药', '商贸零售', '国防军工', '基础化工', '家电',
        '建材', '建筑', '房地产', '有色金属', '机械', '汽车', '消费者服务', '煤炭',
        '电力及公用事业', '电力设备及新能源', '电子', '石油石化', '纺织服装', '综合',
        '综合金融', '计算机', '轻工制造', '通信', '钢铁', '银行', '非银行金融', '食品饮料'
    ]
    
    def __init__(self, universe: pd.DataFrame, frequency: str, start_date: int, end_date: int, deal_price: str = 'preclose') -> None:
        self.dl = DataLoader(save=False)
        self.univ = Universe()
        self.universe = universe
        self.frequency = frequency
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = Calendar()
        self.aligner = Aligner()
        self.trade_dates = self.aligner.trade_dates
        self.tickers = self.aligner.tickers
        self.deal_price = deal_price
        self.rebal_dates = self._get_rebal_dates(self.start_date, self.end_date, self.frequency)
        self._prepare_basic_set()
        self._prepare_risk_model()

    def _prepare_basic_set(self) -> None:
        basic_set = {
            'stock_quote': self.dl.load('stock_quote', keys=list(set(['open', 'close', self.deal_price, 'adjfactor']))),
            'index_quote': self.dl.load('index_quote')
        }
        self.stock_adjopen = basic_set['stock_quote']['open'] * basic_set['stock_quote']['adjfactor']
        self.stock_adjclose = basic_set['stock_quote']['close'] * basic_set['stock_quote']['adjfactor']
        self.stock_close_return = self.stock_adjclose / self.stock_adjclose.shift(1) - 1
        self.stock_open_return  =self.stock_adjopen / self.stock_adjclose.shift(1) - 1
        if self.deal_price in ['open', 'vwap', 'close']:
            self.stock_deal_price = basic_set['stock_quote'][self.deal_price] * basic_set['stock_quote']['adjfactor']
            self.stock_sell_return = self.stock_deal_price / self.stock_adjclose.shift(1) - 1
            self.stock_buy_return = self.stock_adjclose / self.stock_deal_price - 1
        else:
            self.stock_deal_price = self.stock_adjclose.shift(1)

    def _prepare_industry(self, name) -> None:
        AShareIndustriesClassCITICS = self.dl.load('AShareIndustriesClassCITICS')
        info = AShareIndustriesClassCITICS[AShareIndustriesClassCITICS['INDUSTRIESNAME'] == name][['S_INFO_WINDCODE', 'ENTRY_DT', 'REMOVE_DT']]
        df = self.univ._format_universe(self.univ.arrange_info_table(info))
        return df

    def _prepare_risk_model(self) -> None:
        self.risk_model = {}
        
        # 对数总市值
        stock_size = self.dl.load('stock_size')
        lncap = np.log(stock_size['total_mv'] / 1e8)  # 亿元
        lncap = (lncap - lncap.mean(axis=1).values[:, None]) / lncap.std(axis=1).values[:, None]
        md = lncap.median(axis=1)
        mad = (lncap - md.values[:, None]).abs().median(axis=1)
        lower = md.values - 1.483 * 3 * mad.values
        upper = md.values + 1.483 * 3 * mad.values
        lncap = lncap.clip(lower[:, None], upper[:, None], axis=1)
        self.risk_model['lncap'] = self.aligner.align(lncap)

    def _get_rebal_dates(self, start_date: int, end_date: int, frequency: str):
        start_date = max(start_date, self.trade_dates[0])
        end_date = min(end_date, self.trade_dates[-1])
        if isinstance(frequency, int):
            rebal_dates = self.calendar.get_trade_dates_between(start_date, end_date, True, True)[::frequency]
        elif frequency == 'M':
            rebal_dates = self.calendar.month_ends
        elif frequency == 'W':
            rebal_dates = self.calendar.week_ends
        else:
            raise Exception(f"Invalid frequency {frequency}")
        return rebal_dates[(rebal_dates >= start_date) & (rebal_dates <= end_date)]
    
    def _cut_groups(self, f: pd.Series, ngroups):
        N = len(f)
        tks = f.index.values
        n = N // ngroups
        groups = []
        for i in range(ngroups):
            groups.append(tks[n * i : n * (i + 1)])
        return groups
    
    def calc_portfolio_return_and_turnover(self, holding, weight=None):
        rebal_dates = np.array(sorted(list(holding.keys())))
        next_rebal_dates = np.array([self.calendar.get_next_trade_date(i) for i in rebal_dates])
        daily_return = {}
        daily_turnover = {}
        for i in range(len(self.trade_dates)):
            day = self.trade_dates[i]
            if day < rebal_dates[0] or day > self.end_date:
                continue
            if day == rebal_dates[0]:
                daily_return[day] = 0
                daily_turnover[day] = np.nan
                lst_p, lst_w = None, None
                continue
            last_rebal_date = rebal_dates[rebal_dates < day][-1]
            if day in next_rebal_dates:
                # 调仓日
                cur_p = holding[last_rebal_date]
                if weight is None:
                    cur_w = np.ones(len(cur_p)) / len(cur_p)
                else:
                    cur_w = weight[last_rebal_date]
                if day == next_rebal_dates[0]:
                    # 第一个调仓日：以昨收盘价买入
                    ret = self.stock_close_return.loc[day, cur_p].fillna(value=0).values
                    daily_ret = (ret * cur_w).sum()
                    turn = np.nan
                    cur_w = cur_w * (1 + ret)
                    cur_w = cur_w / cur_w.sum()
                else:
                    # 不是第一个调仓日：按照deal_price类型进行调仓
                    cross_p = np.unique(lst_p.tolist() + cur_p.tolist())
                    if self.deal_price == 'preclose':
                        # 以昨收盘价调仓
                        ret = self.stock_close_return.loc[day, cur_p].fillna(value=0).values
                        daily_ret = (ret * cur_w).sum()
                        lst_p2w = dict(zip(lst_p, lst_w))
                        cur_p2w = dict(zip(cur_p, cur_w))
                        turn = sum([np.abs(lst_p2w.get(j, 0) - cur_p2w.get(j, 0)) for j in cross_p])
                        cur_w = cur_w * (1 + ret)
                        cur_w = cur_w / cur_w.sum()
                    else:
                        # 不是以昨收盘价调仓
                        ret_sell = self.stock_sell_return.loc[day, lst_p].fillna(value=0).values
                        daily_ret_sell = (ret_sell * lst_w).sum()
                        ret_buy = self.stock_buy_return.loc[day, cur_p].fillna(value=0).values
                        daily_ret_buy = (ret_buy * cur_w).sum()
                        daily_ret = (1 + daily_ret_sell) * (1 + daily_ret_buy) - 1
                        lst_w = lst_w * (1 + ret_sell)
                        lst_w = lst_w / lst_w.sum()
                        lst_p2w = dict(zip(lst_p, lst_w))
                        cur_p2w = dict(zip(cur_p, cur_w))
                        turn = sum([np.abs(lst_p2w.get(j, 0) - cur_p2w.get(j, 0)) for j in cross_p])
                        cur_w = cur_w * (1 + ret_buy)
                        cur_w = cur_w / cur_w.sum()
                lst_p = cur_p
                lst_w = cur_w
            else:
                # 非调仓日
                ret = self.stock_close_return.loc[day, lst_p].fillna(value=0).values
                daily_ret = (ret * lst_w).sum()
                lst_w = lst_w * (1 + ret)
                lst_w = lst_w / lst_w.sum()
                turn = np.nan
            daily_return[day] = daily_ret
            daily_turnover[day] = turn
        return daily_return, daily_turnover
    
    def calc_stock_forward_returns(self):
        forward_returns = {}
        for i in range(len(self.rebal_dates) - 1):
            ret = self.stock_adjclose.loc[self.rebal_dates[i + 1]] / self.stock_adjclose.loc[self.rebal_dates[i]] - 1
            forward_returns[self.rebal_dates[i]] = ret
        forward_returns = pd.DataFrame(forward_returns).T
        forward_returns.index.name = self.stock_adjclose.index.name
        return forward_returns
    
    def calc_factor_statistics(self, factor):
        forward_returns = self.calc_stock_forward_returns()
        ic_series = {}
        ric_series = {}
        tstats_series = {}
        ic_series_by_ind = {}
        ric_series_by_ind = {}
        for day in self.rebal_dates[:-1]:
            f = factor.loc[day].dropna()
            tks = f.index.values
            f = f.values
            r = forward_returns.loc[day, tks].fillna(0).values
            dat = pd.DataFrame({'factor': f, 'forward_return': r})
            # IC
            ic = dat.corr().iloc[0, 1]
            # Rank IC
            ric = dat.corr(method='spearman').iloc[0, 1]
            # tstats
            tstats = sm.OLS(r, sm.add_constant(f)).fit().tvalues[1] if len(dat) > 1 else np.nan
            # 分行业IC & Rank IC
            ic_by_ind = {}
            ric_by_ind = {}
            for name in self.RISK_INDUSTRY_FACTORS:
                ind = self._prepare_industry(name).loc[day, tks].values
                ic_by_ind[name] = dat.iloc[ind == 1].corr().iloc[0, 1]
                ric_by_ind[name] = dat.iloc[ind == 1].corr(method='spearman').iloc[0, 1]
            ic_series[day] = ic
            ric_series[day] = ric
            tstats_series[day] = tstats
            ic_series_by_ind[day] = ic_by_ind
            ric_series_by_ind[day] = ric_by_ind
        ic_series = pd.Series(ic_series).dropna()
        ric_series = pd.Series(ric_series).dropna()
        tstats_series = pd.Series(tstats_series).dropna()
        ic_series_by_ind = pd.DataFrame(ic_series_by_ind).T.dropna(how='all')
        ric_series_by_ind = pd.DataFrame(ric_series_by_ind).T.dropna(how='all')
        return ic_series, ric_series, tstats_series, ic_series_by_ind, ric_series_by_ind
    
    def calc_factor_style_corr(self, factor):
        # style_corr_series = {}
        # for i, day in enumerate(self.trade_dates):
        #     f = factor.iloc[i]
        #     style_corr = {}
        #     for name in self.RISK_STYLE_FACTORS:
        #         style = self.risk_model[name].iloc[i]
        #         dat = pd.DataFrame({'factor': f, 'style': style}).dropna()
        #         style_corr[name] = dat.corr(method='spearman').iloc[0, 1]
        #     style_corr_series[day] = style_corr
        # style_corr_series = pd.DataFrame(style_corr_series).T.dropna(how='all')
        # return style_corr_series
        return
    
    def get_latest_score_info(self, factor):
        stock_description = self.dl.load('stock_description')
        last_day = factor.dropna(how='all').index.max()
        factor = pd.DataFrame({
            'trade_date': [last_day] * len(self.tickers),
            'ticker': self.tickers,
            'factor': factor.loc[last_day].values
        }).dropna()
        industry_names = []
        for indname in self.RISK_INDUSTRY_FACTORS:
            ind = self._prepare_industry(indname).loc[last_day]
            ind = ind[ind == 1]
            industry_names += np.c_[ind.index.values, [indname] * len(ind)].tolist()
        industry_names = pd.DataFrame(industry_names, columns=['ticker', 'indname'])
        industry_names['ticker'] = industry_names['ticker'].astype(int)
        factor = pd.merge(factor, stock_description[['ticker', 'name']], on=['ticker'])
        factor = pd.merge(factor, industry_names, on=['ticker'])
        factor = factor.sort_values('factor', ascending=False)
        return factor
    
    def calc_factor_performance(self, daily_rets, daily_turns, ic_series, ric_series, tstats_series, ic_series_by_ind, ric_series_by_ind, style_corr_series, latest_score_info):
        group_names = daily_rets.columns.tolist()
        format_group_names = [f'G{i}' for i in range(1, len(group_names) + 1)]
        format_group_names[0] = 'Short'
        format_group_names[-1] = 'Long'
        output = {}
        output['ic_series'] = ic_series
        output['ric_series'] = ric_series
        output['tstats_series'] = tstats_series
        output['groups_return_series'] = daily_rets
        output['groups_nav_series'] = (daily_rets + 1).cumprod()
        output['groups_return_annual'] = (daily_rets - daily_rets.mean(axis=1).values[:, None] + 1).prod() ** (self.ANNUALIZE_MULTIPLIER['D'] / (len(daily_rets) - 1)) - 1
        output['groups_turnover_series'] = daily_turns
        output['lms_return_series'] = daily_rets[group_names[-1]] - daily_rets[group_names[0]]
        output['lms_nav_series'] = (output['lms_return_series'] + 1).cumprod()
        output['lms_drawdown_series'] = output['lms_nav_series'] / output['lms_nav_series'].cummax() - 1
        output['lmm_return_series'] = daily_rets[group_names[-1]] - daily_rets.mean(axis=1)
        output['lmm_nav_series'] = (output['lmm_return_series'] + 1).cumprod()
        output['lmm_drawdown_series'] = output['lmm_nav_series'] / output['lmm_nav_series'].cummax() - 1
        output['ic_series_by_ind'] = ic_series_by_ind
        output['ic_by_ind'] = ic_series_by_ind.mean()
        output['ric_series_by_ind'] = ric_series_by_ind
        output['ric_by_ind'] = ric_series_by_ind.mean()
        # output['style_corr_series'] = style_corr_series
        # output['style_corr'] = style_corr_series.mean()
        output['latest_score_info'] = latest_score_info

        years, year_counts = np.unique(self.rebal_dates // 10000, return_counts=True)
        if year_counts[0] == 1:
            years, year_counts = years[1:], year_counts[1:]
        period_names = [f'{self.rebal_dates[0]} - {self.rebal_dates[-1]}']
        starts = [daily_rets.index[0]]
        ends = [daily_rets.index[-1]]
        states = [1]
        for year in years:
            period_names.append(f'{year}Y')
            starts.append(year * 10000 + 101)
            ends.append(year * 10000 + 1231)
            states.append(0)
        
        performance = {}
        for name, start, end, state in zip(period_names, starts, ends, states):
            _ic_series = ic_series.loc[start:end]
            _ric_series = ric_series.loc[start:end]
            _tstats_series = tstats_series.loc[start:end]
            _groups_turnover_series = output['groups_turnover_series'].loc[start:end]
            _lms_return_series = output['lms_return_series'].loc[start:end]
            _lmm_return_series = output['lmm_return_series'].loc[start:end]
            ic = _ic_series.mean()
            ic_ir = ic / _ic_series.std() * np.sqrt(self.ANNUALIZE_MULTIPLIER[self.frequency])
            ic_win = (_ic_series > 0).mean()
            ric = _ric_series.mean()
            ric_win = (_ric_series > 0).mean()
            ric_ir = ric / _ric_series.std() * np.sqrt(self.ANNUALIZE_MULTIPLIER[self.frequency])
            tstats = _tstats_series.mean()
            if state == 0:
                lms_annual = (_lms_return_series + 1).prod() - 1
                lmm_annual = (_lmm_return_series + 1).prod() - 1
                long_turnover = _groups_turnover_series[group_names[-1]].sum()
            else:
                lms_annual = (_lms_return_series + 1).prod() ** (self.ANNUALIZE_MULTIPLIER['D'] / (len(_lms_return_series) - 1)) - 1
                lmm_annual = (_lmm_return_series + 1).prod() ** (self.ANNUALIZE_MULTIPLIER['D'] / (len(_lmm_return_series) - 1)) - 1
                long_turnover = _groups_turnover_series[group_names[-1]].sum() / (_groups_turnover_series[group_names[-1]].count() - 1) * self.ANNUALIZE_MULTIPLIER[self.frequency]
            lms_sharpe = _lms_return_series.mean() / _lms_return_series.std() * np.sqrt(self.ANNUALIZE_MULTIPLIER['D'])
            lmm_sharpe = _lmm_return_series.mean() / _lmm_return_series.std() * np.sqrt(self.ANNUALIZE_MULTIPLIER['D'])
            lms_nav_series = (_lms_return_series + 1).cumprod()
            lmm_nav_series = (_lmm_return_series + 1).cumprod()
            lms_max_drawdown = (lms_nav_series / lms_nav_series.cummax() - 1).min()
            lmm_max_drawdown = (lmm_nav_series / lmm_nav_series.cummax() - 1).min()
            lms_win = (_lms_return_series > 0).mean()
            lmm_win = (_lmm_return_series > 0).mean()
            performance[name] = {
                'ic': ic,
                'ric': ric,
                'ic_ir': ic_ir,
                'ric_ir': ric_ir,
                'tstats': tstats,
                'ic_win': ic_win,
                'ric_win': ric_win,
                'lms_annual': lms_annual,
                'lms_sharpe': lms_sharpe,
                'lms_max_drawdown': lms_max_drawdown,
                'lms_win': lms_win,
                'lmm_annual': lmm_annual,
                'lmm_sharpe': lmm_sharpe,
                'lmm_max_drawdown': lmm_max_drawdown,
                'lmm_win': lmm_win,
                'long_turnover': long_turnover,
            }
        performance = pd.DataFrame(performance)
        performance = performance[period_names[1:] + period_names[:1]]
        performance = performance.T
        output['performance'] = performance

        output['ic_series'].index.name = ''
        output['ic_series'].index = pd.to_datetime(output['ic_series'].index.astype(str))
        output['ic_series'].name = 'ic'

        output['ric_series'].index.name = ''
        output['ric_series'].index = pd.to_datetime(output['ric_series'].index.astype(str))
        output['ric_series'].name = 'ric'

        output['tstats_series'].index.name = ''
        output['tstats_series'].index = pd.to_datetime(output['tstats_series'].index.astype(str))
        output['tstats_series'].name = 'tstats'

        output['groups_return_series'].index.name = ''
        output['groups_return_series'].index = pd.to_datetime(output['groups_return_series'].index.astype(str))
        output['groups_return_series'].columns = format_group_names

        output['groups_return_annual'].index.name = '分组'
        output['groups_return_annual'].index = format_group_names
        output['groups_return_annual'].name = 'annual_return'

        output['groups_turnover_series'].index.name = ''
        output['groups_turnover_series'].index = pd.to_datetime(output['groups_turnover_series'].index.astype(str))
        output['groups_turnover_series'].columns = format_group_names

        output['groups_nav_series'].index.name = ''
        output['groups_nav_series'].index = pd.to_datetime(output['groups_nav_series'].index.astype(str))
        output['groups_nav_series'].columns = format_group_names

        output['lms_return_series'].index.name = ''
        output['lms_return_series'].index = pd.to_datetime(output['lms_return_series'].index.astype(str))
        output['lms_return_series'].name = 'lms_return'

        output['lms_nav_series'].index.name = ''
        output['lms_nav_series'].index = pd.to_datetime(output['lms_nav_series'].index.astype(str))
        output['lms_nav_series'].name = 'lms_nav'

        output['lms_drawdown_series'].index.name = ''
        output['lms_drawdown_series'].index = pd.to_datetime(output['lms_drawdown_series'].index.astype(str))
        output['lms_drawdown_series'].name = 'lms_drawdown'

        output['lmm_return_series'].index.name = ''
        output['lmm_return_series'].index = pd.to_datetime(output['lmm_return_series'].index.astype(str))
        output['lmm_return_series'].name = 'lmm_return'

        output['lmm_nav_series'].index.name = ''
        output['lmm_nav_series'].index = pd.to_datetime(output['lmm_nav_series'].index.astype(str))
        output['lmm_nav_series'].name = 'lmm_nav'

        output['lmm_drawdown_series'].index.name = ''
        output['lmm_drawdown_series'].index = pd.to_datetime(output['lmm_drawdown_series'].index.astype(str))
        output['lmm_drawdown_series'].name = 'lmm_drawdown'

        output['ic_series_by_ind'].index.name = ''
        output['ic_series_by_ind'].index = pd.to_datetime(output['ic_series_by_ind'].index.astype(str))

        output['ic_by_ind'].index.name = '行业'
        output['ic_by_ind'].name = 'ic_by_ind'

        output['ric_series_by_ind'].index.name = ''
        output['ric_series_by_ind'].index = pd.to_datetime(output['ric_series_by_ind'].index.astype(str))

        output['ric_by_ind'].index.name = '行业'
        output['ric_by_ind'].name = 'ric_by_ind'

        # output['style_corr_series'].index.name = ''
        # output['style_corr_series'].index = pd.to_datetime(output['style_corr_series'].index.astype(str))

        # output['style_corr'].index.name = '风格'
        # output['style_corr'].name = 'style_corr'

        output['performance'].index.name = '日期'

        return output
    
    def figure(self, output, factor_name):
        # 分组净值（折线图）
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        output['groups_nav_series'].plot(ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'Nav by Group ({factor_name})')

        # 分组净值Demean（折线图）
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        data = output['groups_return_series']
        data = data - data.mean(axis=1).values[:, None]
        data = (1 + data).cumprod()
        data.plot(ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'Nav by Group (demean) ({factor_name})')

        # 分组收益Demean（柱状图）
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        output['groups_return_annual'].plot(ax=ax, kind='bar')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'Annual Return by Group (demean) ({factor_name})')

        # 多空净值（折线图）
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        output['lms_nav_series'].plot(ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'Long-Short Nav ({factor_name})')

        # 多头超额净值（折线图）
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        output['lmm_nav_series'].plot(ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'Long-Benchmark Nav ({factor_name})')

        # 分行业IC（柱状图）
        fig, ax = plt.subplots(figsize=(18, 9), dpi=150)
        output['ic_by_ind'].plot(ax=ax, kind='bar')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'IC by Industy ({factor_name})')
        
        # # 风格因子相关系数
        # fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        # output['style_corr'].plot(ax=ax, kind='bar')
        # ax.set_xlabel('')
        # ax.set_ylabel('')
        # ax.set_title(f'Correlation Coeffificents with Style Factors ({factor_name})')
                    
    def test(self, factor: pd.DataFrame, ngroups: int = 10, display: bool = False, factor_name: str = None):
        factor = self.aligner.align(factor)
        factor_name = 'factor' if factor_name is None else factor_name

        time0 = time.time()
        print('-' * 100)
        print(f'Start testing for factor [{factor_name}] ...')

        factor = factor.copy()
        factor[self.universe.astype(int) == 0] = np.nan

        # 分组
        holdings = {i: {} for i in range(1, ngroups + 1)}
        for day in self.rebal_dates:
            ix = np.where(self.trade_dates == day)[0][0]
            f = factor.iloc[ix].dropna().sort_values()
            groups = self._cut_groups(f, ngroups)
            for i in range(1, ngroups + 1):
                holdings[i][day] = groups[i - 1]
        
        # 分组回测（日度收益率 & 日度换手率）
        daily_rets = {}
        daily_turns = {}
        for i in range(1, ngroups + 1):
            daily_rets[i], daily_turns[i] = self.calc_portfolio_return_and_turnover(holdings[i])
        daily_rets = pd.DataFrame(daily_rets).dropna()
        daily_turns = pd.DataFrame(daily_turns).dropna()

        # 因子回测 (IC & ICIR & tstats & 分行业IC)
        ic_series, ric_series, tstats_series, ic_series_by_ind, ric_series_by_ind = self.calc_factor_statistics(factor)

        # 风格因子相关系数
        style_corr_series = self.calc_factor_style_corr(factor)

        # 最新一期因子值
        latest_score_info = self.get_latest_score_info(factor)

        # 绩效统计
        output = self.calc_factor_performance(daily_rets, daily_turns, ic_series, ric_series, tstats_series, ic_series_by_ind, ric_series_by_ind, style_corr_series, latest_score_info)

        # 画图
        if display:
            self.figure(output, factor_name)

        time1 = time.time()
        print(f'Total time elapsed = {time1 - time0:.2f} seconds')

        return output