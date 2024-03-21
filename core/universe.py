'''
Author: WangXiang
Date: 2024-03-21 21:05:04
LastEditTime: 2024-03-21 21:08:47
'''

import numpy as np
import pandas as pd
from bisect import bisect_left, bisect_right

from ..conf import PATH_DATA_BASIC_PROCESSED
from .aligner import Aligner
from .loader import DataLoader
from .tools import format_unstack_table


class Universe:

    def __init__(self) -> None:
        self.aligner = Aligner()
        self.trade_dates = self.aligner.trade_dates
        self.info_table = {}
        self.info_list = ['AShareST', 'AShareEODPrices', 'AShareDescription', 'AIndexMembers', 'AShareIndustriesClassCITICS', 'AShareSWNIndustriesClass']
        dl = DataLoader(save=False)
        for table in self.info_list:
            self.info_table[table] = dl.load(table)
        self.aligner = Aligner()

    def arrange_info_table(self, info_table: pd.DataFrame) -> pd.DataFrame:
        assert info_table.shape[1] == 3
        fisrt_day, last_day = 19000101, 21000101
        df = info_table.copy()
        df.columns = ['ticker', 'entry_date', 'remove_date']
        df['remove_date'] = df['remove_date'].fillna(value=last_day)
        df[['entry_date', 'remove_date']] = df[['entry_date', 'remove_date']].astype(int)
        tickers = np.sort(np.unique(df['ticker']))
        tickers_ix = dict(zip(tickers, np.arange(len(tickers))))
        data = np.zeros((len(self.trade_dates), len(tickers)))
        for i in range(len(df)):
            ticker, entry_date, remove_date = df.iloc[i]
            start = bisect_left(self.trade_dates, entry_date)
            end = bisect_right(self.trade_dates, remove_date)
            data[slice(start, end), tickers_ix[ticker]] = 1
        return pd.DataFrame(data, index=self.trade_dates, columns=tickers)
    
    def _format_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.aligner.align(format_unstack_table(df).fillna(value=0)).fillna(value=0)
    
    def handle_listed_days(self, listed_days: int = 365):
        listed_info = self.info_table['AShareDescription'][['S_INFO_WINDCODE', 'S_INFO_LISTDATE', 'S_INFO_DELISTDATE']].copy()
        entry_date_shift = pd.to_datetime(listed_info['S_INFO_LISTDATE']) + pd.DateOffset(days=listed_days)
        listed_info['S_INFO_LISTDATE'] = entry_date_shift.dt.strftime('%Y%m%d')
        return self._format_universe(self.arrange_info_table(listed_info))
    
    def handle_discontinuous_trade(self, window: int = 20) -> pd.DataFrame:
        zero_trade_info = self.info_table['AShareEODPrices'][['S_DQ_LOW', 'S_DQ_AMOUNT', 'S_DQ_VOLUME']]
        valid_trade_sign = (zero_trade_info.S_DQ_LOW > 1) & (zero_trade_info.S_DQ_AMOUNT > 1) * (zero_trade_info.S_DQ_VOLUME > 1)
        untrade_samples = valid_trade_sign.loc[~valid_trade_sign].astype(int)
        untrade_samples[:] = 1
        untrade_samples = untrade_samples.unstack()
        untrade_samples.index = untrade_samples.index.astype(int)
        result = untrade_samples.fillna(0).rolling(window).sum()
        result = (result > 0)
        return self._format_universe(result)
    
    def is_st(self) -> pd.DataFrame:
        st_info = self.info_table['AShareST'].copy()
        st_info = st_info[st_info['S_TYPE_ST'] != 'R'][['S_INFO_WINDCODE', 'ENTRY_DT', 'REMOVE_DT']]
        return self._format_universe(self.arrange_info_table(st_info))
    
    def is_suspended(self) -> pd.DataFrame:
        suspend_info = self.info_table['AShareEODPrices'].S_DQ_TRADESTATUSCODE
        suspend_info = suspend_info[suspend_info == 0]
        suspend_info[:] = 1
        return self._format_universe(suspend_info.unstack())
    
    def is_price_limit(self) -> pd.DataFrame:
        price_limit_info = self.info_table['AShareEODPrices'][['S_DQ_CLOSE', 'S_DQ_LIMIT', 'S_DQ_STOPPING']]
        result = price_limit_info[(price_limit_info['S_DQ_CLOSE'] >= price_limit_info['S_DQ_LIMIT']) | (price_limit_info['S_DQ_CLOSE'] <= price_limit_info['S_DQ_STOPPING'])].copy().S_DQ_CLOSE
        result[:] = 1
        return self._format_universe(result.unstack()).shift(-1).fillna(value=1).astype(int)
    
    def handle_index_member(self, index_code: str):
        index_member_info = self.info_table['AIndexMembers'].copy()
        index_member_info = index_member_info[index_member_info['S_INFO_WINDCODE'] == index_code][['S_CON_WINDCODE', 'S_CON_INDATE', 'S_CON_OUTDATE']]
        return self._format_universe(self.arrange_info_table(index_member_info))
    
    def __call__(
            self,
            listed_days:           int  = 0,
            continuous_trade_days: int  = 0,
            include_st:            bool = True,
            include_suspend:       bool = True,
            include_price_limit:   bool = True,
            index_code:            str  = None,
    ):
        """
        Filter securities based on specified conditions.

        Parameters
        ----------
        listed_days : int, optional
            Number of days since listing. Default is 0.
        continuous_trade_days : int, optional
            Number of consecutive trade days. Default is 0.
        include_st : bool, optional
            Include ST securities. Default is True.
        include_suspend : bool, optional
            Include suspended securities. Default is True.
        include_price_limit : bool, optional
            Include securities with price limit. Default is True.
        index_code : str, optional
            Keep the securites which are members of the index. Default is None.
        """
        # Filter based on listed days
        univ = self.handle_listed_days(listed_days)
        
        # Filter out ST securities if include_st is set to False
        if include_st is False:
            univ *= (self.is_st() == 0)
        
        # Filter out suspended securities if include_suspend is set to False
        if include_suspend is False:
            univ *= (self.is_suspended() == 0)
        
        # Filter out securities with price limit if include_price_limit is set to False
        if include_price_limit is False:
            univ *= (self.is_price_limit() == 0)
        
        # Filter based on consecutive trade days
        if continuous_trade_days > 0:
            univ *= (self.handle_discontinuous_trade(continuous_trade_days) == 0)

        # Filter based on index code
        if index_code is not None:
            univ *= (self.handle_index_member(index_code) == 1)
        
        return univ.astype(int)
    

if __name__ == "__main__":
    univ = Universe()
    market_univ = univ(listed_days=120, continuous_trade_days=20, include_st=False, include_suspend=False, include_price_limit=False)
    hs300_univ = univ(listed_days=120, continuous_trade_days=20, include_st=False, include_suspend=False, include_price_limit=False, index_code='000300.SH')
    