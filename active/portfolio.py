'''
Author: WangXiang
Date: 2024-03-31 10:18:30
LastEditTime: 2024-04-11 19:28:30
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple, Union

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

from .. import conf
from ..core import Aligner, Calendar, DataLoader, Universe


class ActivePortfolio:

    def __init__(self, start_date: int, end_date: int, deal_price: str = 'preclose') -> None:
        self.dl = DataLoader(save=False)
        self.univ = Universe()
        self.calendar = Calendar()
        self.aligner = Aligner()
        self.init_universe = self.univ()
        self.start_date = start_date
        self.end_date = end_date
        self.trade_dates = self.aligner.trade_dates
        self.tickers = self.aligner.tickers
        self.deal_price = deal_price
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
    
    def _get_rebal_dates(self, start_date: int, end_date: int, frequency: str):
        start_date = max(start_date, self.trade_dates[0])
        end_date = min(end_date, self.trade_dates[-1])
        if isinstance(frequency, int):
            rebal_dates = self.calendar.get_trade_dates_between(start_date, end_date, True, True)[::frequency]
        elif frequency == 'M':
            rebal_dates = self.calendar.month_ends
        elif frequency == 'W':
            rebal_dates = self.calendar.week_ends
        elif frequency == 'Q':
            rebal_dates = self.calendar.month_ends
            rebal_dates = rebal_dates[rebal_dates // 100 % 100 % 3 == 0]
        else:
            raise Exception(f"Invalid frequency {frequency}")
        return rebal_dates[(rebal_dates >= start_date) & (rebal_dates < end_date)]

    def calc_portfolio_daily_return(
        self,
        factors: Dict[str, Tuple[pd.DataFrame, Union[int, float, Callable]]],
        universe: Dict[str, pd.DataFrame],
        weights: Dict[str, pd.DataFrame],
        frequency: str,
        cost: float = 0.0015,
        **kwargs
    ) -> Tuple[pd.Series, Tuple[Dict[int, List[int]], Dict[int, List[float]]]]:
        assert len(weights) == 1
        weight_name = list(weights.keys())[0]
        rebal_dates = self._get_rebal_dates(self.start_date, self.end_date, frequency)
        for factor_name, (factor, threshold) in factors.items():
            factor = factor.copy()
            factor[self.init_universe == 0] = np.nan
            for u in universe.values():
                factor[u.astype(int) == 0] = np.nan
            factor = factor.loc[rebal_dates]
            factors[factor_name] = (factor, threshold)

        portfolio_name_factors = ''
        for factor_name, (factor, threshold) in factors.items():
            if portfolio_name_factors != '':
                portfolio_name_factors += '_'
            if isinstance(threshold, int):
                if threshold > 0:
                    portfolio_name_factors += f'{factor_name}_S{threshold}'
                else:
                    portfolio_name_factors += f'{factor_name}_L{abs(threshold)}'
            elif isinstance(threshold, float):
                if abs(threshold) >= 1:
                    if threshold > 0:
                        portfolio_name_factors += f'{factor_name}_S{int(threshold)}'
                    else:
                        portfolio_name_factors += f'{factor_name}_L{abs(int(threshold))}'
                else:
                    if threshold > 0:
                        portfolio_name_factors += f'{factor_name}_S{threshold * 100:.2f}%'
                    else:
                        portfolio_name_factors += f'{factor_name}_L{abs(threshold * 100):.2f}%'
        portfolio_name_universe = '_'.join([f'{name}' for name in universe.keys()])
        portfolio_name_weights = f'{weight_name}'
        portfolio_name = '{}, U_{}, W_{}, {}'.format(portfolio_name_factors, portfolio_name_universe, portfolio_name_weights, frequency)

        holding = {}
        weight = {}
        for day in rebal_dates:
            h = None
            for factor_name, (factor, threshold) in factors.items():
                hh = factor.loc[day].dropna().sort_values()
                if isinstance(threshold, int):
                    if threshold > 0:
                        hh = hh.index[:threshold].values
                    else:
                        hh = hh.index[threshold:].values
                elif isinstance(threshold, float):
                    if abs(threshold) >= 1:
                        if threshold > 0:
                            hh = hh.index[:int(threshold)].values
                        else:
                            hh = hh.index[int(threshold):].values
                    else:
                        if threshold > 0:
                            hh = hh.index[:max(1, int(len(hh) * threshold))].values
                        else:
                            hh = hh.index[min(-1, int(len(hh) * threshold)):].values
                elif isinstance(threshold, Callable):
                    hh = threshold(hh, day=day, **kwargs)
                if h is None:
                    h = hh
                else:
                    hh = np.intersect1d(h, hh)
                    if len(hh) >= 10:  # 组合至少要有的持股数量
                        h = hh
                    else:
                        print(f'In portfolio [{portfolio_name}] ...\nAt day [{day}], filter of [{factor_name}] is skipped otherwise will lead to empty portfolio.\n', '-'*60)
            holding[day] = h
            if weights[weight_name] is None:
                w = np.ones_like(h) / len(h)  # equal
            else:
                w = weights[weight_name].loc[day, h].fillna(value=0).values
                if w.sum() == 0:
                    print(f'In portfolio [{portfolio_name}] ...\nAat day [{day}], original weights are all zeros, set the final weights to equal.', '-'*60)
                    w = np.ones_like(w) / len(w)
                else:
                    w = w / w.sum()
            weight[day] = w
        daily_return, daily_turnover = self.calc_portfolio_return_and_turnover(holding, weight)
        daily_return = pd.Series(daily_return, name=portfolio_name)
        daily_turnover = pd.Series(daily_turnover, name=portfolio_name)
        return daily_return - cost * daily_turnover.fillna(value=0), (holding, weight)
    
    def figure(
        self,
        active_portfolios: pd.DataFrame,
        active_bmk_names: List[str],
        index_bmk_codes: List[str] = None,
        frequency: str = None,
        active_bmk_portfolios: pd.DataFrame = None,
        index_bmk_names: List[str] = None,
        with_excess_names: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        active_portfolios = active_portfolios.copy()
        for name in active_bmk_names:
            assert name in active_portfolios.columns
        if active_bmk_portfolios is not None:
            assert len(active_portfolios) == len(active_bmk_portfolios)
            active_bmk_portfolios = active_bmk_portfolios[np.setdiff1d(active_bmk_portfolios.columns.values, active_portfolios.columns.values)]
            active_portfolios = pd.concat([active_portfolios, active_bmk_portfolios], axis=1).sort_index()
            active_bmk_names += active_bmk_portfolios.columns.tolist()
        bmk_cols = active_bmk_names.copy()
        if index_bmk_codes is not None:
            AIndexEODPrices = self.dl.load('AIndexEODPrices')
            AWindIndexEODPrices = self.dl.load('AWindIndexEODPrices')
            for i in range(len(index_bmk_codes)):
                code = index_bmk_codes[i]
                if code[-2:] == 'WI':
                    ret = AWindIndexEODPrices.loc(axis=0)[:, code]['S_DQ_PCTCHANGE'].droplevel(1) / 100
                else:
                    ret = AIndexEODPrices.loc(axis=0)[:, code]['S_DQ_PCTCHANGE'].droplevel(1) / 100
                ret.index = ret.index.astype(int)
                ret.index.name = active_portfolios.index.name
                ret = ret.reindex(index=active_portfolios.index)
                ret.iloc[0] = 0
                name = code if index_bmk_names is None else index_bmk_names[i]
                active_portfolios[name] = ret
                bmk_cols.append(name)
        if with_excess_names is not None:
            if with_excess_names is True:
                with_excess_names = np.setdiff1d(active_portfolios.columns.values, bmk_cols)
                print(with_excess_names)
            for name in with_excess_names:
                for col in bmk_cols:
                    active_portfolios[f'{name}, E_{col}'] = (active_portfolios[name] + 1) / (active_portfolios[col] + 1) - 1
        data = (active_portfolios + 1).cumprod()
        if frequency is not None:
            slice_dates = self._get_rebal_dates(self.start_date, self.end_date, frequency)
            slice_dates = np.intersect1d(data.index.values, slice_dates)
            data = data.loc[slice_dates]
        data.index = pd.to_datetime(data.index.astype(str))
        figsize = kwargs.get('figsize', (12, 8))
        dpi = kwargs.get('dpi', 150)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        data.plot(ax=ax)
        if kwargs.get('title', None) is not None:
            ax.set_title(kwargs['title'])
        if kwargs.get('grid', None) is not None:
            ax.grid(kwargs['grid'])
        return data
    
    def calc_performance(self, portfolios: pd.DataFrame, multiplier: int = 252) -> pd.DataFrame:
        """
        portfolios: NAV
        年化收益，年化波动，年化夏普比率，最大回撤
        """
        ann_ret = portfolios.iloc[-1] ** (multiplier / (len(portfolios) - 1)) - 1
        ann_vol = (portfolios / portfolios.shift(1) - 1).std() * np.sqrt(multiplier)
        ann_sharpe = ann_ret / ann_vol
        max_drawdown = (portfolios / portfolios.cummax() - 1).min()
        monthly_nav = portfolios.loc[np.intersect1d(pd.to_datetime(self.calendar.month_ends.astype(str)), portfolios.index)]
        monthly_ret = (monthly_nav / monthly_nav.shift(1) - 1).iloc[1:]
        monthly_win = (monthly_ret >= 0).mean()
        performance = pd.DataFrame({
            '年化收益': ann_ret,
            '年化波动': ann_vol,
            '夏普比率': ann_sharpe,
            '月度胜率': monthly_win,
            '最大回撤': max_drawdown
        })
        return performance
    
    def calc_performance_by_year(self, portfolios: pd.DataFrame, multiplier: int = 252) -> pd.DataFrame:
        start_date = pd.to_datetime(str(portfolios.index.min()))
        end_date = pd.to_datetime(str(portfolios.index.max()))
        years = list(range(start_date.year, end_date.year + 1))
        if len(self.calendar.get_trade_dates_between(int(start_date.strftime('%Y%m%d')), start_date.year * 10000 + 1231, include_start=True, include_end=True)) == 1:
            years = years[1:]
        result = {'年化收益': {}, '年化波动': {}, '夏普比率': {}, '月度胜率': {}, '最大回撤': {}}
        for year in years:
            stt, end = (year - 1) * 10000 + 1231, year * 10000 + 1231
            stt = max(int(start_date.strftime('%Y%m%d')), stt)
            if not self.calendar.is_trade_date(stt):
                stt = self.calendar.get_prev_trade_date(stt)
            end = min(int(end_date.strftime('%Y%m%d')), end)
            if not self.calendar.is_trade_date(end):
                end = self.calendar.get_prev_trade_date(end)
            if stt not in self.calendar.year_ends or end not in self.calendar.year_ends:
                need_to_annualize = True
                if stt in self.calendar.year_ends:
                    year_str = f'{year}(至{end % 10000:04d})'
                else:
                    year_str = f'{year}(从{stt % 10000:04d})'
            else:
                need_to_annualize = False
                year_str = str(year)
            stt = pd.to_datetime(str(stt))
            end = pd.to_datetime(str(end))
            port_nav = portfolios.loc[stt:end]
            port_ret = (portfolios / portfolios.shift(1) - 1).dropna().loc[stt:end]
            ann_ret = port_nav.iloc[-1] / port_nav.iloc[0] - 1
            ann_vol = port_ret.std() * np.sqrt(multiplier)
            if need_to_annualize:
                ann_sharpe = ((ann_ret + 1) ** (multiplier / len(port_nav)) - 1) / ann_vol
            else:
                ann_sharpe = ann_ret / ann_vol
            max_drawdown = (port_nav / port_nav.cummax() - 1).min()
            monthly_nav = portfolios.loc[np.intersect1d(pd.to_datetime(self.calendar.month_ends.astype(str)), port_nav.index)]
            monthly_ret = (monthly_nav / monthly_nav.shift(1) - 1).iloc[1:]
            monthly_win = (monthly_ret >= 0).mean()
            result['年化收益'][year_str] = ann_ret
            result['年化波动'][year_str] = ann_vol
            result['夏普比率'][year_str] = ann_sharpe
            result['月度胜率'][year_str] = monthly_win
            result['最大回撤'][year_str] = max_drawdown
        result = {k: pd.DataFrame(v).T for k, v in result.items()}
        return result