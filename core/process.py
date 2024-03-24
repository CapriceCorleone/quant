'''
Author: WangXiang
Date: 2024-03-24 17:29:32
LastEditTime: 2024-03-24 18:42:51
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit

from .data_processor import DataProcessTask
from .factor_processor import AnalystFactorProcessor
from .njit.analyst import afunc_consensus


class AShareConsensus(DataProcessTask):
    """
    一致预期计算，窗口期及半衰期由类属性params定义，默认为122天及21天（交易日）
    """
    def __init__(self) -> None:
        super().__init__()
        self.task = 'incremental'
        self.field_date = 'trade_date'
        self.params = {
            'window': 122,
            'half_life': 21,
        }
        self.field_earning_est = 'EST_NET_PROFIT'
        self.field_financial = 'NET_PROFIT_EXCL_MIN_INT_INC'

    def operator(self, earning_est, report, calendar, init_date):
        """
        单元处理函数
        :param earning_est: 个股历史盈利预测序列
        :param report: 个股年报净利润
        :param calendar: 交易日期序列
        :param init_date: 起始日期
        :return: np.array: 个股自init_date起的一致预期
        """
        return afunc_consensus(earning_est, report, calendar, init_date, **self.params)

    def run(self, AShareEarningEst, AShareIncome, init_date):
        """
        一致预期处理主函数，调用AnalystFactorProcessor.
        :param AShareEarningEst: Wind盈利预测明细表
        :param AShareIncome: Wind利润表
        :param init_date: 计算起始日期
        :return pd.DataFrame: 一致预期数据，包含交易日，股票代码，上期年报日期，上期年报业绩，fy0，fy1，fy2，fttm（未来12个月）一致预期业绩
        """
        afp = AnalystFactorProcessor()
        earning_est_col = ['S_INFO_WINDCODE', 'EST_DT', 'RESEARCH_INST_NAME', 'REPORTING_PERIOD'] + [self.field_earning_est]
        earning_est = AShareEarningEst.loc[AShareEarningEst['EST_DT'] >= str(init_date - 20000)][earning_est_col].copy()
        earning_est[self.field_earning_est] = earning_est[self.field_earning_est] * 10000
        # convert inst_name to numeric id
        earning_est['RESEARCH_INST_NAME'] = afp.convert_to_id(earning_est['RESEARCH_INST_NAME'])
        # convert dtypes
        int_col = ['EST_DT', 'REPORTING_PERIOD']
        earning_est[int_col] = earning_est[int_col].astype(int)
        # select annual report
        income_col = ['S_INFO_WINDCODE', 'ANN_DT', 'REPORT_PERIOD'] + [self.field_financial]
        annual_report = AShareIncome.loc[(AShareIncome['REPORT_PERIOD'].astype(int) % 10000 == 1231) & (AShareIncome['STATEMENT_TYPE'] == '408001000')][income_col]
        annual_report = annual_report.loc[annual_report['S_INFO_WINDCODE'].isin(earning_est['S_INFO_WINDCODE'])]
        # split data by tickers
        split_earning_est = afp.spilt_by_ticker(earning_est.values)
        split_annual_report = afp.spilt_by_ticker(annual_report.values)
        ticker_list = sorted(set.intersection(set(split_earning_est.keys()), set(split_annual_report.keys())))
        result_by_ticker = {}
        # calculate by tickers
        if len(afp.trade_days[afp.trade_days >= init_date]) > 244:
            ticker_list = tqdm(ticker_list, ncols=80, desc=self.operator.__name__)
        for code in ticker_list:
            result_by_ticker[code] = self.operator(split_earning_est[code], split_annual_report[code], afp.trade_days, init_date)
        # stack data
        codes = [np.tile(code, len(result_by_ticker[code])) for code in result_by_ticker]
        data = np.vstack(list(result_by_ticker.values()))
        data[:, 2:] = data[:, 2:]
        df = pd.DataFrame({
            'trade_date': data[:, 0],
            'ticker': np.hstack(codes),
            'last_period': data[:, 1],
            'lr': data[:, 2],
            'fy0': data[:, 3],
            'fy1': data[:, 4],
            'fy2': data[:, 5],
            'fttm': data[:, 6],
        })
        df[['trade_date', 'last_period']] = df[['trade_date', 'last_period']].astype(int)
        df = df.sort_values(['trade_date', 'ticker'])
        return df
    

class FamaFrench3Factor(DataProcessTask):

    def __init__(self) -> None:
        self.task = 'incremental'
        self.field_date = 'trade_date'
    
    @staticmethod
    @njit
    def rank_label(x, quantiles):
        y = np.empty(len(x), dtype=np.float32)
        for i in range(1, len(quantiles)):
            y[(x > np.nanpercentile(x, quantiles[i - 1] * 100)) & (x <= np.nanpercentile(x, quantiles[i] * 100))] = i
        y[x == np.nanmin(x)] = 1
        y[np.isnan(x)] = np.nan
        return y
    
    def get_rank_return(self, df_stock_return, df_indicator, bins):
        """
        按指标分组
        """
        df_indicator = df_indicator.reindex_like(df_stock_return)
        df_label = [self.rank_label(zz, bins) for zz in df_indicator.values]
        df_label = pd.DataFrame(df_label, index=df_indicator.index, columns=df_indicator.columns)
        group_returns = {}
        for i in range(1, len(bins)):
            group_returns[i] = df_stock_return[df_label == i].mean(1)
        group_returns = pd.DataFrame(group_returns)
        long_short_pnl = group_returns.iloc[:, 0] - group_returns.iloc[:, -1]
        return long_short_pnl
    
    def run(self, AShareEODPrices, AShareEODDerivativeIndicator, AIndexEODPrices, init_date):
        # 过滤未满一年股票
        stock_return = AShareEODPrices['S_DQ_PCTCHANGE'].loc[str(init_date - 20000):].unstack('S_INFO_WINDCODE')
        invalid_ix = stock_return.shift(244).isna()
        invalid_ix.iloc[:244] = False
        stock_return[invalid_ix] = np.nan
        stock_return = stock_return.loc[str(init_date):]
        stock_return[abs(stock_return) > 0.2] = np.nan
        # 计算分组指标
        indicators = AShareEODDerivativeIndicator[['S_DQ_MV', 'S_VAL_PB_NEW']].loc[str(init_date - 10000):].unstack('S_INFO_WINDCODE').shift(1).loc[str(init_date):]
        # 计算收益
        smb = self.get_rank_return(stock_return, indicators['S_DQ_MV'], np.array([0.0, 0.5, 1.0]))
        hml = self.get_rank_return(stock_return, 1 / indicators['S_VAL_PB_NEW'], np.array([0.0, 0.3, 0.7, 1.0]))
        mkt = AIndexEODPrices.loc(axis=0)[:, '000002.SH']['S_DQ_PCTCHANGE'].droplevel(1, 0).loc[str(init_date):]
        # 合并输出
        df = pd.concat([hml, smb, mkt], axis=1).sort_index()
        df.index = df.index.astype(int)
        df = df.loc[init_date:].reset_index()
        df.columns = ['trade_date', 'HML', 'SML', 'MKT']
        return df.dropna()