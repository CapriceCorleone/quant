'''
Author: WangXiang
Date: 2024-03-21 21:13:43
LastEditTime: 2024-03-21 21:23:26
'''

import numpy as np
import pandas as pd
from datetime import datetime

from .loader import DataLoader


MIN_TRADE_DATE = 19000101
MAX_TRADE_DATE = 21000101


class Calendar:

    def __init__(self) -> None:
        self.dl = DataLoader(save=False)
        AShareCalendarAll = self.dl.load('AShareCalendarAll')
        trade_dates = np.sort(AShareCalendarAll.TRADE_DAYS.astype(int).unique())
        tds_datetime = [datetime(i // 10000, i // 100 % 100, i % 100) for i in trade_dates]
        self.trade_dates = trade_dates
        self.year = np.array([i.year for i in tds_datetime])
        self.month = np.array([i.month for i in tds_datetime])
        self.day = np.array([i.day for i in tds_datetime])
        self.week = np.array([i.isocalendar()[1] for i in tds_datetime])  # 年初前几个交易日和前一年年末的交易日如果处于同一周，则计数时仍按照上一年的计数
        self.weeday = np.array([i.isocalendar()[2] for i in tds_datetime])  # 周一 ~ 周日对应 1~7
        self.is_year_end = np.zeros_like(trade_dates, dtype=int)
        self.is_month_end = np.zeros_like(trade_dates, dtype=int)
        self.is_week_end = np.zeros_like(trade_dates, dtype=int)
        for i in range(len(trade_dates)):
            if i == len(trade_dates) - 1:
                continue
            if self.year[i + 1] != self.year[i]:
                self.is_year_end[i] = 1
            if self.month[i + 1] != self.month[i]:
                self.is_month_end[i] = 1
            if self.week[i + 1] != self.week[i]:
                self.is_week_end[i] = 1
        
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            'trade_date':   self.trade_dates,
            'year':         self.year,
            'month':        self.month,
            'day':          self.day,
            'week':         self.week,
            'weekday':      self.weeday,
            'is_year_end':  self.is_year_end,
            'is_month_end': self.is_month_end,
            'is_week_end':  self.is_week_end,
        })

    def get_prev_trade_date(self, day: int, n = 1) -> int:
        tgt_dates = self.trade_dates[self.trade_dates < day]
        if len(tgt_dates) < n:
            return MIN_TRADE_DATE
        return tgt_dates[-n]
    
    def get_next_trade_date(self, day: int, n = 1) -> int:
        tgt_dates = self.trade_dates[self.trade_dates > day]
        if len(tgt_dates) < n:
            return MAX_TRADE_DATE
        return tgt_dates[n - 1]
    
    def is_trade_date(self, day: int) -> bool:
        return day in self.trade_dates
    
    def get_trade_dates_between(self, start_date: int, end_date: int, include_start: bool = True, include_end: bool = True):
        if include_start:
            if include_end:
                return self.trade_dates[(self.trade_dates >= start_date) & (self.trade_dates <= end_date)]
            return self.trade_dates[(self.trade_dates >= start_date) & (self.trade_dates < end_date)]
        else:
            if include_end:
                return self.trade_dates[(self.trade_dates > start_date) & (self.trade_dates <= end_date)]
            return self.trade_dates[(self.trade_dates > start_date) & (self.trade_dates < end_date)]

    @property
    def month_ends(self):
        return self.trade_dates[self.is_month_end == 1]
    
    @property
    def week_ends(self):
        return self.trade_dates[self.is_week_end == 1]
    
    @property
    def year_ends(self):
        return self.trade_dates[self.is_year_end == 1]