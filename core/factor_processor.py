'''
Author: WangXiang
Date: 2024-03-24 17:29:24
LastEditTime: 2024-03-26 20:40:39
'''

import numpy as np
import pandas as pd
from numba import njit

from .loader import DataLoader


@njit
def fill_to_date(m, date):
    """将数据按照date序列进行填充，m第一列为公告日期"""
    data = np.zeros((len(date), m.shape[1])) * np.nan
    data[:, 0] = date
    for n in range(len(m)):
        if m[n, 0] <= date[0]:
            data[:, 1:] = m[n, 1:]
        else:
            ix_start = np.where(date < m[n, 0])[0].max()  # 数据对齐到公布日前一交易日
            ix_end = np.where(date < m[n, 0] + 10000)[0].max() + 1
            data[ix_start : ix_end, 1:] = m[n, 1:]
    return data[data[:, 1] >= data[:, 0] - 10000]


class FundamentalFactorProcessor:

    def __init__(self) -> None:
        dl = DataLoader()
        self.trade_days = np.sort(dl.load('AShareCalendar')['TRADE_DAYS'].values.astype(int))
        self.fixed_fields = {
            'ticker': 'S_INFO_WINDCODE',  # 股票ticker字段
            'ann_dt': 'ANN_DT',  # 公告日字段
            'report_period': 'REPORT_PERIOD'  # 财报日期字段
        }
        stock_description = dl.load('AShareDescription')
        self.ticker = stock_description['S_INFO_WINDCODE'].tolist()
    
    @classmethod
    def spilt_by_ticker(cls, data):
        """将data按照第一列分为dict"""
        def double_sort(r):
            return r[np.lexsort((r[:, 1], r[:, 0]))]
        # spilt financial data by ticker
        data = data[~np.isnan(data[:, -1].astype(float))]
        data = data[data[:, 0].argsort()]  # sort by col-0 if not already sorted
        stamp = np.r_[0, np.flatnonzero(data[1:, 0] > data[:-1, 0]) + 1, data.shape[0]]
        tickers = data[stamp[:-1], 0].astype(str)
        return {
            tickers[i]: double_sort(data[stamp[i]: stamp[i + 1], 1:]).astype(np.float64)
            for i in range(len(stamp) - 1)
        }
    
    @classmethod
    def concat_by_ticker(cls, split_data):
        """将dict数据合并为DataFrame"""
        codes = [np.tile(code, len(split_data[code])) for code in split_data]
        data = np.vstack(list(split_data.values()))
        return pd.DataFrame({
            'trade_date': data[:, 0],
            'ticker': np.hstack(codes),
            'factor': data[:, -1]
        })
    
    def process(self, financials, items, operator, init_date, align_date = True, return_factor = True):
        """
        因子基本计算函数-纯基本面因子
        :param financials: list of DataFrame, 包含一个或多个DataFrame财务报表数据的列表
        :param items: List of strings, 包含一个或多个财务报表科目名称，与df_financials对应
        :param operator: 财务数据运算函数
        :param init_date: 起始计算日
        :param return_factor: 是否返回DataFrame，否则为dict
        :param align_date: 是否对齐到交易日
        :return: 基本面因子
        """
        # 将财务数据按照股票代码拆分为dict
        spilt_financials = []
        for n in range(len(financials)):
            assert items[n] in financials[n].columns
            # 选择目标字段（固定字段+目标科目）
            n_slice = financials[n][list(self.fixed_fields.values()) + [items[n]]].copy()
            # 日期转化为int
            int_columns = [self.fixed_fields['ann_dt'], self.fixed_fields['report_period']]
            n_slice[int_columns] = n_slice[int_columns].astype(int)
            # 选择近5年数据
            n_slice = n_slice.loc[n_slice[self.fixed_fields['ann_dt']].values > init_date - 50000]
            # 按ticker拆分
            spilt_financials.append(self.spilt_by_ticker(n_slice.values))
        # 获取不同数据股票代码交集
        ticker_list = [set(_.keys()) for _ in spilt_financials]
        inter_ticker = sorted(set.intersection(*ticker_list))
        # 将operator应用到每个股票的数据，数据格式通常为[日期，财报日期，数据]
        result_by_ticker = {
            code: operator(*[f[code] for f in spilt_financials]) for code in inter_ticker
        }
        
        if align_date:
            # 数据对齐到交易日
            target_dates = self.trade_days[self.trade_days >= init_date]
            result_by_ticker = {
                code: fill_to_date(result_by_ticker[code], target_dates) for code in inter_ticker
            }
        
        if return_factor:
            # 返回DataFrame
            factor = self.concat_by_ticker(result_by_ticker).dropna()
            factor['trade_date'] = factor['trade_date'].astype(np.int32)
            return factor
        else:
            # 返回dict
            return result_by_ticker
    

class AnalystFactorProcessor(FundamentalFactorProcessor):
    """
    处理AShareEarningEst等分析师表数据
    """
    def __init__(self) -> None:
        super(AnalystFactorProcessor, self).__init__()
        self.fixed_fields = {
            'ticker': 'S_INFO_WINDCODE',  # 股票ticker
            'ann_dt': 'ANN_DT',  # 报告发布日
            'class_name': 'RESEARCH_INST_NAME',  # 数据分类字段（按机构或分析师）
            'report_period': 'REPORTING_PERIOD'  # 财报日期
        }
    
    @staticmethod
    def convert_to_id(arg):
        # 将str转化为int格式的ID
        id_dict = {x: id(x) for x in arg.unique()}
        return arg.replace(id_dict)
    
    def process(self, analyst_date, item, operator, init_date, return_factor=True):
        assert item in analyst_date.columns
        # 抓取目标数据
        data = analyst_date.loc[analyst_date[self.fixed_fields['ann_dt']] >= str(init_date - 20000)][list(self.fixed_fields.values()) + [item]].copy()
        # 转换机构名为ID
        data[self.fixed_fields['class_name']] = self.convert_to_id(data[self.fixed_fields['class_name']])
        # 日期数据转化为int
        int_columns = [self.fixed_fields['ann_dt'], self.fixed_fields['report_period']]
        data[int_columns] = data[int_columns].astype(int)
        # 将数据按股票代码拆分为dict
        data = self.spilt_by_ticker(data.values)
        # 将operator应用到每个股票的数据，数据通常为[日期，机构，财报日期，数据]
        result_by_ticker = {
            code: operator(data[code], self.trade_days, init_date) for code in data.keys()
        }
        if return_factor:
            factor = self.concat_by_ticker(result_by_ticker).dropna()
            factor['trade_date'] = factor['trade_date'].astype(np.int32)
            return factor
        else:
            return result_by_ticker


class ValueFactorProcessor(FundamentalFactorProcessor):

    def process(self, divide_by, financials, items, operator, init_date):
        """
        :param divide_by: pd.DataFrame格式：市值等处于分母位置的变量，index为时间，columns为股票
        :param financials: 财务表
        :param items: 财务表中对应的字段
        :param operator: 字段处理方法
        :param init_date: 起始日期
        :return: pd.DataFrame格式因子
        """
        # 计算财务指标
        financial_df = super(ValueFactorProcessor, self).process(financials, items, operator, init_date)
        financial_df = financial_df.set_index(['trade_date', 'ticker'])['factor'].unstack()
        # 计算估值因子
        divide_by = divide_by.loc[init_date:]
        factor_df = financial_df.reindex_like(divide_by) / divide_by
        # 转换为列数据
        factor = factor_df.stack().reset_index()
        factor.columns = ['trade_date', 'ticker', 'factor']
        return factor
    

class StaticProcessors(type):
    """
    创建processor示例，供因子计算调用，避免重复实例化
    """
    @property
    def value(self):
        return ValueFactorProcessor()
    
    @property
    def analyst(self):
        return AnalystFactorProcessor()
    
    @property
    def fundamental(self):
        return FundamentalFactorProcessor()
    

class Processors(metaclass=StaticProcessors):
    pass