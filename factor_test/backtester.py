'''
Author: WangXiang
Date: 2024-03-20 22:36:50
LastEditTime: 2024-03-21 22:49:55
'''

import os
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, List

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

    RISK_INDUSTRY_FACTORS = [
        '交通运输', '传媒', '农林牧渔', '医药', '商贸零售', '国防军工', '基础化工', '家电',
        '建材', '建筑', '房地产', '有色金属', '机械', '汽车', '消费者服务', '煤炭',
        '电力及公用事业', '电力设备及新能源', '电子', '石油石化', '纺织服装', '综合',
        '综合金融', '计算机', '轻工制造', '通信', '钢铁', '银行', '非银行金融', '食品饮料'
    ]
    
    def __init__(self, universe: Universe, frequency: str, start_date: int, end_date: int, deal_price: str = 'close') -> None:
        self.dl = DataLoader(save=False)
        self.universe = universe
        self.calendar = Calendar()
        self.aligner = Aligner()
        self.trade_dates = self.aligner.trade_dates
        self.tickers = self.aligner.tickers
        self.deal_price = deal_price
        self._prepare_basic_set()
        self._prepare_risk_model()

    def _prepare_basic_set(self) -> None:
        self.basic_set = {
            'stock_quote': self.dl.load('stock_quote'),
            'index_quote': self.dl.load('index_quote')
        }

    def _prepare_risk_model(self) -> None:
        self.risk_model = {}

        # 中信一级行业
        univ = Universe()
        AShareIndustriesClassCITICS = self.dl.load('AShareIndustriesClassCITICS')
        for name in self.RISK_INDUSTRY_FACTORS:
            info = AShareIndustriesClassCITICS[AShareIndustriesClassCITICS['INDUSTRIESNAME'] == name][['S_INFO_WINDCODE', 'ENTRY_DT', 'REMOVE_DT']]
            df = univ._format_universe(self.univ.arrange_info_table(info))
            self.risk_model[name] = df
        
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

    def _get_rebal_dates(self, start_date, end_date, frequency):
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
    
    def _cut_groups(self, f, ngroups):
        N = len(f)
        tks = f.index.values
        n = N // ngroups
        groups = []
        for i in range(ngroups):
            groups.append(tks[n * i : n * (i + 1)])
        return groups

    def calc_portfolio_return_and_turnover(self, holding, weight):
        rebal_dates = np.array(sorted(list(holding.keys())))
        next_rebal_dates = np.array([self.calendar.get_next_trade_date(i) for i in rebal_dates])
        daily_return = {}
        daily_turnover = {}
        for i in range(len(self.trade_dates)):
            day = self.trade_dates[i]
            if day < rebal_dates[0] or day > rebal_dates[-1]:
                continue
            if day == rebal_dates[0]:
                daily_return[day] = 0
                daily_turnover[day] = np.nan
                lst_p, lst_w = None, None
                continue
            last_rebal_date = rebal_dates[rebal_dates < day][-1]
            if day in next_rebal_dates:
                cur_p = holding[last_rebal_date]
                cur_w = weight[last_rebal_date]
                stock_ret = self.stock_daily_returns.loc[day, cur_p].fillna(0).values  # TODO
                ret = (stock_ret * cur_w).sum()
                if lst_p is None:
                    turn = np.nan
                else:
                    joint_tks = np.unique(lst_p.tolist() + cur_p.tolist())
                    lst_pw = dict(zip(lst_p, lst_w))
                    cur_pw = dict(zip(cur_p, cur_w))
                    turn = sum([np.abs(lst_pw.get(j, 0) - cur_pw.get(j, 0)) for j in joint_tks])
                cur_w = cur_w * (1 + stock_ret)
                cur_w = cur_w / cur_w.sum()
                lst_p = cur_p
                lst_w = cur_w
            else:
                stock_ret = self.stock_daily_returns.loc[day, lst_p].fillna(0).values
                ret = (stock_ret * lst_w).sum()
                turn = np.nan
            daily_return[day] = ret
            daily_turnover[day] = turn
        return daily_return, daily_turnover