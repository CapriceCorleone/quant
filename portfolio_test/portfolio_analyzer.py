'''
Author: WangXiang
Date: 2024-04-09 20:12:07
LastEditTime: 2024-04-14 04:05:48
'''

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from .. import conf
from ..conf import variables as V
from ..core import DataLoader, Calendar, Aligner, Universe


class PortfolioAnalyzer:

    ANNUALIZE_MULTIPLIER = V.ANNUALIZE_MULTIPLIER
    RISK_STYLE_FACTORS = V.RISK_STYLE_FACTORS
    RISK_INDUSTRY_FACTORS = V.RISK_INDUSTRY_FACTORS

    def __init__(self, deal_price: str = 'vwap') -> None:
        self.dl = DataLoader(save=False)
        self.risk_model_dl = DataLoader(save=False)
        self.risk_model_dl.root = conf.PATH_RISK_MODEL_DATA
        self.calendar = Calendar()
        self.aligner = Aligner()
        self.univ = Universe()
        self.trade_dates = self.aligner.trade_dates
        self.tickers = self.aligner.tickers
        self.deal_price = deal_price
        self._prepare_basic_set()
        self._prepare_risk_model()

    def _prepare_basic_set(self) -> None:
        basic_set = {
            'stock_quote': self.dl.load('stock_quote', keys=list(set(['open', 'close', self.deal_price, 'adjfactor', 'amount']))),
            'index_quote': self.dl.load('index_quote')
        }
        self.stock_adjopen = basic_set['stock_quote']['open'] * basic_set['stock_quote']['adjfactor']
        self.stock_adjclose = basic_set['stock_quote']['close'] * basic_set['stock_quote']['adjfactor']
        self.stock_close_return = self.stock_adjclose / self.stock_adjclose.shift(1) - 1
        self.stock_open_return = self.stock_adjopen / self.stock_adjclose.shift(1) - 1
        if self.deal_price in ['open', 'vwap', 'close']:
            self.stock_deal_price = basic_set['stock_quote'][self.deal_price] * basic_set['stock_quote']['adjfactor']
            self.stock_sell_return = self.stock_deal_price / self.stock_adjclose.shift(1) - 1
            self.stock_buy_return = self.stock_adjclose / self.stock_deal_price - 1
        else:
            self.stock_deal_price = self.stock_adjclose.shift(1)
        self.stock_amount = basic_set['stock_quote']['amount'] / 100000

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

        # style
        for name in self.RISK_STYLE_FACTORS:
            df = self.risk_model_dl.load(name)
            self.risk_model[name] = self.aligner.align(df)

    def convert_portfolio_dataframe(self, portfolio: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
        if portfolio.shape[1] == 3:
            portfolio.columns = ['trade_date', 'ticker', 'weight']
        elif portfolio.shape[1] == 2:
            portfolio.columns = ['trade_date', 'ticker']
            portfolio['weight'] = portfolio.groupby('trade_date')['ticker'].transform(lambda x: 1 / len(x))
        portfolio = portfolio.sort_values(['trade_date', 'ticker']).reset_index(drop=True)
        endix = portfolio.groupby('trade_date')['ticker'].count().cumsum()
        sttix = endix.shift(1).fillna(value=0).astype(int)
        holding = {}
        weight = {}
        for i in range(len(endix)):
            port = portfolio.iloc[sttix.iloc[i]:endix.iloc[i]]
            day = port['trade_date'].iloc[0]
            holding[day] = port['ticker'].values
            weight[day] = port['weight'].values
        return holding, weight
    
    def calc_portfolio_return_and_turnover(self, holding: Dict[int, List[int]], weight: Dict[int, List[float]] = None):
        rebal_dates = np.array(sorted(list(holding.keys())))
        next_rebal_dates = np.array([self.calendar.get_next_trade_date(i) for i in rebal_dates])
        daily_return = {}
        daily_turnover = {}
        for i in range(len(self.trade_dates)):
            day = self.trade_dates[i]
            # if day < rebal_dates[0] or day > self.end_date:
            if day < rebal_dates[0]:  # PortfolioAnalyzer中暂时不指定end_date，组合收益率计算至最新一天
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
    
    def calc_index_return(self, index_code: str, start_date: int):
        AIndexEODPrices = self.dl.load('AIndexEODPrices')
        index_return = AIndexEODPrices.loc(axis=0)[:, index_code]['S_DQ_PCTCHANGE'].droplevel(1) / 100
        index_return.index = index_return.index.astype(int)
        index_return = index_return.loc[start_date:]
        index_return.iloc[0] = 0
        return index_return
    
    def calc_daily_return_statistics(self, daily_return: pd.Series) -> Dict[str, Any]:
        daily_value = (1 + daily_return).cumprod()
        monthly_value = daily_value.loc[np.intersect1d(daily_value.index, self.calendar.month_ends)].sort_index()
        monthly_return = (monthly_value / monthly_value.shift(1) - 1).dropna()
        # 总收益
        total_return = (1 + daily_return).prod() - 1
        # 年化收益
        annual_return = (1 + total_return) ** (self.ANNUALIZE_MULTIPLIER['D'] / (len(daily_return) - 1)) - 1
        # 回撤
        drawdown = daily_value / daily_value.cummax() - 1
        # 最大回撤
        max_drawdown = min(drawdown)
        # 日胜率
        daily_win_rate = (daily_return.iloc[1:] > 0).mean()
        # 月胜率
        monthly_win_rate = (monthly_return > 0).mean()
        # 波动率 (跟踪误差)
        volatility = daily_return.std() * np.sqrt(self.ANNUALIZE_MULTIPLIER['D'])
        tracking_error = volatility
        # 夏普比率 (信息比率)
        sharpe_ratio = annual_return / volatility
        information_ratio = sharpe_ratio
        # 回撤区间 & 回撤幅度
        drawdown_periods = [[]]
        for i in range(len(drawdown)):
            if drawdown.iloc[i] >= 0:
                if len(drawdown_periods[-1]) != 0:
                    drawdown_periods.append([])
                continue
            drawdown_periods[-1].append(drawdown.index[i])
        if len(drawdown_periods[-1]) == 0:
            drawdown_periods.pop(-1)
        drawdown_magnitude = [daily_value.loc[p].min() / daily_value.loc[:p[0]].max() - 1 for p in drawdown_periods]
        drawdown_magnitude = np.array(drawdown_magnitude)
        # 最大回撤区间
        max_drawdown_day = drawdown.index[drawdown.argmin()]
        for p in drawdown_periods:
            if max_drawdown_day in p:
                max_drawdown_period = f'{p[0]}-{p[-1]}'
                break
        # 单日回撤0.5%天数占比
        large_daily_drawdown_ratio = (daily_return.iloc[1:] < -0.005).mean() if daily_return.iloc[1:].min() < -0.005 else 0
        # 滚动60日最大回撤中位数
        average_max_drawdown_rolling_3m = (daily_value.rolling(60, min_periods=1).apply(lambda x: (x / x.cummax() - 1).min())).median()
        # 回撤突破1%平均幅度
        average_large_drawdown = drawdown_magnitude[drawdown_magnitude < -0.01].mean() if drawdown_magnitude.min() < -0.01 else 0
        # 回撤突破1%次数
        num_large_drawdown = (drawdown_magnitude < -0.01).sum() if drawdown_magnitude.min() < -0.01 else 0
        # 回撤突破1%持续天数
        lasting_days_large_drawdown = np.mean([len(p) for p, m in zip(drawdown_periods, drawdown_magnitude) if m < -0.01]) if drawdown_magnitude.min() < -0.01 else 0
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'daily_win_rate': daily_win_rate,
            'monthly_win_rate': monthly_win_rate,
            'volatility': volatility,
            'tracking_error': tracking_error,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'drawdown_periods': [f'{p[0]}-{p[-1]}' for p in drawdown_periods],
            'drawdown_magnitude': drawdown_magnitude.tolist(),
            'max_drawdonw_period': max_drawdown_period,
            'large_daily_drawdown_ratio': large_daily_drawdown_ratio,
            'average_max_drawdown_rolling_3m': average_max_drawdown_rolling_3m,
            'average_large_drawdown': average_large_drawdown,
            'num_large_drawdown': num_large_drawdown,
            'lasting_days_large_drawdown': lasting_days_large_drawdown,
        }
    
    def calc_daily_turnover_statistics(self, daily_turnover: pd.Series) -> Dict[str, Any]:
        average_turnover = daily_turnover[daily_turnover != 0].mean()
        turnover_counts = daily_turnover[daily_turnover != 0].groupby(daily_turnover.index // 10000).count().median(0)
        annual_turnover = average_turnover * turnover_counts
        return {
            'average_turnover': average_turnover,
            'annual_turnover': annual_turnover
        }
    
    def calc_performance(
        self,
        portfolio: pd.DataFrame        = None,
        holding: Dict[int, List[int]]  = None,
        weight: Dict[int, List[float]] = None,
        index_code: str                = None,
        cost: float                    = 0.0015,
        deep: bool                     = False
    ) -> pd.DataFrame:
        
        performance = {}

        if portfolio is not None:  # 可以传入形如trade_date, ticker或trade_date, ticker, weight的dataframe表示持仓
            holding, weight = self.convert_portfolio_dataframe(portfolio)
        start_date = min(list(holding.keys()))
        
        # 计算组合日度收益率和日度换手率
        daily_return, daily_turnover = self.calc_portfolio_return_and_turnover(holding, weight)
        daily_return = pd.Series(daily_return)
        daily_turnover = pd.Series(daily_turnover)
        daily_return = daily_return - cost * daily_turnover.fillna(value=0)

        # 计算基准指数日度收益率
        if index_code is None:
            index_return = daily_return.copy() * 0
        else:
            index_return = self.calc_index_return(index_code, start_date)
            index_return = index_return.reindex_like(daily_return)
        
        # 计算相对基准指数的超额收益率
        excess_return = (1 + daily_return) / (1 + index_return) - 1

        # 计算全局统计指标
        daily_return_stats = self.calc_daily_return_statistics(daily_return)
        if index_code is not None:
            index_return_stats = self.calc_daily_return_statistics(index_return)
        else:
            index_return_stats = None
        excess_return_stats = self.calc_daily_return_statistics(excess_return)
        daily_turnover_stats = self.calc_daily_turnover_statistics(daily_turnover)
        overall_stats = {
            '绝对收益': daily_return_stats['annual_return'],
            '基准收益': index_return_stats['annual_return'] if index_return_stats is not None else 0.0,
            '超额收益': excess_return_stats['annual_return'],
            '最大回撤': excess_return_stats['max_drawdown'],
            '日胜率': excess_return_stats['daily_win_rate'],
            '月胜率': excess_return_stats['monthly_win_rate'],
            '跟踪误差': excess_return_stats['tracking_error'],
            '换手率/次': daily_turnover_stats['average_turnover'],
            '换手率/年': daily_turnover_stats['annual_turnover'],
            '信息比率': excess_return_stats['information_ratio'],
            '夏普比率': daily_return_stats['sharpe_ratio'],
            '最大回撤时间': excess_return_stats['max_drawdonw_period'],
            '单日回撤0.5%天数占比': excess_return_stats['large_daily_drawdown_ratio'],
            '滚动60日最大回撤中位数': excess_return_stats['average_max_drawdown_rolling_3m'],
            '回撤突破1%平均幅度': excess_return_stats['average_large_drawdown'],
            '回撤突破1%次数': excess_return_stats['num_large_drawdown'],
            '回撤突破1%持续天数': excess_return_stats['lasting_days_large_drawdown'],
            '绝对收益回撤记录': pd.DataFrame({'回撤区间': daily_return_stats['drawdown_periods'], '回撤幅度': daily_return_stats['drawdown_magnitude']}),
            '超额收益回撤记录': pd.DataFrame({'回撤区间': excess_return_stats['drawdown_periods'], '回撤幅度': excess_return_stats['drawdown_magnitude']}),
        }
        performance['年化平均'] = overall_stats

        # 计算分年度统计指标
        counts_by_year = daily_return.groupby(daily_return.index // 10000).count()
        if counts_by_year.iloc[0] == 1:
            years = counts_by_year.index[1:].values
        else:
            years = counts_by_year.index.values
        for year in years:
            stt = self.calendar.get_prev_trade_date(year * 10000 + 101)
            end = year * 10000 + 1231
            daily_ret = daily_return.loc[stt:end].copy()
            index_ret = index_return.loc[stt:end].copy()
            excess_ret = excess_return.loc[stt:end].copy()
            daily_ret.iloc[0] = 0
            index_ret.iloc[0] = 0
            excess_ret.iloc[0] = 0
            daily_turn = daily_turnover.loc[year * 10000 + 101:end]
            daily_return_stats = self.calc_daily_return_statistics(daily_ret)
            if index_code is not None:
                index_return_stats = self.calc_daily_return_statistics(index_ret)
            else:
                index_return_stats = None
            excess_return_stats = self.calc_daily_return_statistics(excess_ret)
            daily_turnover_stats = self.calc_daily_turnover_statistics(daily_turn)
            stats = {
                '绝对收益': daily_return_stats['total_return'],
                '基准收益': index_return_stats['total_return'] if index_return_stats is not None else 0.0,
                '超额收益': excess_return_stats['total_return'],
                '最大回撤': excess_return_stats['max_drawdown'],
                '日胜率': excess_return_stats['daily_win_rate'],
                '月胜率': excess_return_stats['monthly_win_rate'],
                '跟踪误差': excess_return_stats['tracking_error'],
                '换手率/次': daily_turnover_stats['average_turnover'],
                '换手率/年': daily_turnover_stats['annual_turnover'],
                '信息比率': excess_return_stats['information_ratio'],
                '夏普比率': daily_return_stats['sharpe_ratio'],
                '最大回撤时间': excess_return_stats['max_drawdonw_period'],
                '单日回撤0.5%天数占比': excess_return_stats['large_daily_drawdown_ratio'],
                '滚动60日最大回撤中位数': excess_return_stats['average_max_drawdown_rolling_3m'],
                '回撤突破1%平均幅度': excess_return_stats['average_large_drawdown'],
                '回撤突破1%次数': excess_return_stats['num_large_drawdown'],
                '回撤突破1%持续天数': excess_return_stats['lasting_days_large_drawdown'],
                '绝对收益回撤记录': pd.DataFrame({'回撤区间': daily_return_stats['drawdown_periods'], '回撤幅度': daily_return_stats['drawdown_magnitude']}),
                '超额收益回撤记录': pd.DataFrame({'回撤区间': excess_return_stats['drawdown_periods'], '回撤幅度': excess_return_stats['drawdown_magnitude']}),
            }
            performance[year] = stats
        
        performance = pd.DataFrame(performance)
        if deep:
            items = [
                '绝对收益', '基准收益', '超额收益', '最大回撤', '日胜率', '月胜率', '跟踪误差',
                '换手率/次', '换手率/年', '信息比率', '夏普比率', '最大回撤时间',
                '单日回撤0.5%天数占比', '滚动60日最大回撤中位数', '回撤突破1%平均幅度',
                '回撤突破1%次数', '回撤突破1%持续天数', '绝对收益回撤记录', '超额收益回撤记录'
            ]
            return performance.loc[items]
        else:
            items = [
                '绝对收益', '基准收益', '超额收益', '最大回撤', '日胜率', '月胜率', '跟踪误差',
                '换手率/次', '换手率/年', '信息比率', '夏普比率'
            ]
            return performance.loc[items].astype(np.float64)
    
    P = calc_performance

    def calc_portfolio_exposure(self, holding: Dict[int, List[int]], weight: Dict[int, List[float]] = None, adjust_weight: bool = True):
        rebal_dates = np.array(sorted(list(holding.keys())))
        next_rebal_dates = np.array([self.calendar.get_next_trade_date(i) for i in rebal_dates])
        industry = {}
        for name in self.RISK_INDUSTRY_FACTORS:
            ind = self._prepare_industry(name).loc[min(rebal_dates):]
            industry[name] = ind
        daily_exposure = {}
        from tqdm import tqdm
        for i in tqdm(range(len(self.trade_dates)), ncols=80):
            day = self.trade_dates[i]
            if day < rebal_dates[0]:
                continue
            if day == rebal_dates[0]:
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
                if adjust_weight:
                    if day == next_rebal_dates[0]:
                        # 第一个调仓日：以昨收盘价买入
                        ret = self.stock_close_return.loc[day, cur_p].fillna(value=0).values
                        cur_w = cur_w * (1 + ret)
                        cur_w = cur_w / cur_w.sum()
                    else:
                        # 不是第一个调仓日：按照deal_price类型进行调仓
                        cross_p = np.unique(lst_p.tolist() + cur_p.tolist())
                        if self.deal_price == 'preclose':
                            # 以昨收盘价调仓
                            ret = self.stock_close_return.loc[day, cur_p].fillna(value=0).values
                            cur_w = cur_w * (1 + ret)
                            cur_w = cur_w / cur_w.sum()
                        else:
                            # 不是以昨收盘价调仓
                            ret_sell = self.stock_sell_return.loc[day, lst_p].fillna(value=0).values
                            ret_buy = self.stock_buy_return.loc[day, cur_p].fillna(value=0).values
                            lst_w = lst_w * (1 + ret_sell)
                            lst_w = lst_w / lst_w.sum()
                            cur_w = cur_w * (1 + ret_buy)
                            cur_w = cur_w / cur_w.sum()
                lst_p = cur_p
                lst_w = cur_w
            else:
                # 非调仓日
                if adjust_weight:
                    ret = self.stock_close_return.loc[day, lst_p].fillna(value=0).values
                    lst_w = lst_w * (1 + ret)
                    lst_w = lst_w / lst_w.sum()
            daily_exp = []
            for name in self.RISK_STYLE_FACTORS:
                exp = (self.risk_model[name].loc[day, lst_p].fillna(value=0) * lst_w).sum()
                daily_exp.append(exp)
            for name in self.RISK_INDUSTRY_FACTORS:
                exp = (industry[name].loc[day, lst_p].fillna(value=0) * lst_w).sum()
                daily_exp.append(exp)
            daily_exposure[day] = daily_exp
        daily_exposure = pd.DataFrame(daily_exposure, index=self.RISK_STYLE_FACTORS + self.RISK_INDUSTRY_FACTORS).T
        return daily_exposure

    def calc_attribution(
        self,
        portfolio: pd.DataFrame        = None,
        holding: Dict[int, List[int]]  = None,
        weight: Dict[int, List[float]] = None,
        index_code: str                = None,
    ) -> pd.DataFrame:
        
        result = {}
        
        if portfolio is not None:  # 可以传入形如trade_date, ticker或trade_date, ticker, weight的dataframe表示持仓
            holding, weight = self.convert_portfolio_dataframe(portfolio)
        start_date = min(list(holding.keys()))

        # 若指定index_code，则获取index_code对应的持仓和权重
        if index_code is not None:
            if index_code == '000300.SH':
                AIndexHS300CloseWeight = self.dl.load('AIndexHS300CloseWeight')
                portfolio = AIndexHS300CloseWeight[['TRADE_DT', 'S_CON_WINDCODE', 'I_WEIGHT']]
            else:
                AIndexHS300FreeWeight = self.dl.load('AIndexHS300FreeWeight')
                portfolio = AIndexHS300FreeWeight.loc(axis=0)[:, index_code].reset_index()
                portfolio = portfolio[['TRADE_DT', 'S_CON_WINDCODE', 'I_WEIGHT']]
            portfolio['TRADE_DT'] = portfolio['TRADE_DT'].astype(int)
            portfolio = portfolio[portfolio['TRADE_DT'] >= start_date]
            portfolio = portfolio[portfolio['S_CON_WINDCODE'].str[:1].isin(list('036'))]
            portfolio['S_CON_WINDCODE'] = portfolio['S_CON_WINDCODE'].str[:6].astype(int)
            portfolio['I_WEIGHT'] = portfolio.groupby('TRADE_DT')['I_WEIGHT'].transform(lambda x: x / x.sum())
            holding_index, weight_index = self.convert_portfolio_dataframe(portfolio)

        # 组合风格和行业暴露
        portf_exposure = self.calc_portfolio_exposure(holding, weight, adjust_weight=True)
        if index_code is not None:
            index_exposure = self.calc_portfolio_exposure(holding_index, weight_index, adjust_weight=False)
            index_exposure = index_exposure.reindex_like(portf_exposure)
        else:
            index_exposure = portf_exposure * 0.0
        excess_exposure = portf_exposure - index_exposure
        result['exposure'] = {
            'portfolio': portf_exposure,
            'basis': index_exposure,
            'excess': excess_exposure
        }

        # 收益归因
        # TODO

        # 个股超额收益贡献
        # TODO

        return result
    
    A = calc_attribution

    def calc_capacity(
        self,
        portfolio: pd.DataFrame        = None,
        holding: Dict[int, List[int]]  = None,
        weight: Dict[int, List[float]] = None, 
    ):
        if portfolio is not None:  # 可以传入形如trade_date, ticker或trade_date, ticker, weight的dataframe表示持仓
            holding, weight = self.convert_portfolio_dataframe(portfolio)
        rebal_dates = np.array(sorted(list(holding.keys())))
        next_rebal_dates = np.array([self.calendar.get_next_trade_date(i) for i in rebal_dates])
        daily_amount = []
        for i in range(len(self.trade_dates)):
            day = self.trade_dates[i]
            if day <= rebal_dates[0]:
                continue
            last_rebal_date = rebal_dates[rebal_dates < day][-1]
            if day in next_rebal_dates:
                # 调仓日
                cur_p = holding[last_rebal_date]
                if weight is None:
                    cur_w = np.ones(len(cur_p)) / len(cur_p)
                else:
                    cur_w = weight[last_rebal_date]
                cur_amount = self.stock_amount.loc[day, cur_p].values
            else:
                # 非调仓日
                continue
            daily_amount.append([day, cur_amount.sum(), (cur_amount * cur_w).sum(),
                                 cur_amount.min(), cur_w[cur_amount.argmin()],
                                 cur_amount.max(), cur_w[cur_amount.argmax()],
                                 cur_w.min(), cur_amount[cur_w.argmin()],
                                 cur_w.max(), cur_amount[cur_w.argmax()]])
        daily_amount = pd.DataFrame(daily_amount, columns=['trade_date', 'total_amount', 'avg_amount', 'min_amount', 'min_amount_weight', 'max_amount', 'max_amount_weight', 'min_weight', 'min_weight_amount', 'max_weight', 'max_weight_amount'])
        daily_amount = daily_amount.set_index('trade_date').astype(np.float64)
        daily_amount.index = pd.to_datetime(daily_amount.index.astype(str))
        daily_amount.insert(0, 'est_capacity', daily_amount['total_amount'] * 0.1)  # 假设交易占比10%
        return daily_amount