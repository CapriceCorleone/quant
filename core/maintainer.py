'''
Author: WangXiang
Date: 2024-03-21 20:42:06
LastEditTime: 2024-04-14 16:10:55
'''

import os
import pickle
import numpy as np
import pandas as pd
from bisect import bisect_left

from .. import conf
from .loader import DataLoader
from .aligner import Aligner
from .tools import format_unstack_table


class DataMaintainer:

    def __init__(self, roll_back: int = 60) -> None:
        self.roll_back = roll_back
        self.dl = DataLoader()
        self.update_index()
        self.aligner = Aligner()
    
    def update_index(self) -> None:
        AShareEODPrices = self.dl.load('AShareEODPrices')
        trade_dates = np.sort(AShareEODPrices.index.levels[0].astype(int).values)
        tickers = np.sort(AShareEODPrices.index.levels[1].values)
        tickers = np.array([int(i[:6]) for i in tickers if i[0] in list('036')])
        os.makedirs(Aligner.index_path.parent, exist_ok=True)
        np.save(Aligner.index_path, {'trade_date': trade_dates, 'ticker': tickers})
        self.aligner = Aligner()
    
    def update_stock_description(self) -> None:
        AShareDescription = self.dl.load('AShareDescription')
        stock_description = AShareDescription[['S_INFO_WINDCODE', 'S_INFO_NAME', 'S_INFO_LISTDATE', 'S_INFO_DELISTDATE']]
        stock_description.columns = ['ticker', 'name', 'list_date', 'delist_date']
        stock_description = stock_description[stock_description['ticker'].str[0].str.isdigit()]
        stock_description['ticker'] = stock_description['ticker'].str[:6].astype(int)
        stock_description = stock_description[stock_description['ticker'].isin(self.aligner.tickers)]
        stock_description['list_date'] = stock_description['list_date'].fillna(value=19000101).astype(int)
        stock_description['delist_date'] = stock_description['delist_date'].fillna(value=21000101).astype(int)
        stock_description.to_pickle(conf.PATH_DATA_BASIC_PROCESSED / 'stock_description.pkl')

    def update_stock_quote(self, init_date: int = None) -> None:
        if init_date is None:
            init_date = self.aligner.trade_dates[-self.roll_back]
        else:
            init_date = self.aligner.trade_dates[bisect_left(self.aligner.trade_dates[:-self.roll_back], init_date)]
        AShareEODPrices = self.dl.load('AShareEODPrices').loc[str(init_date):]
        stock_quote = {
            'open':      AShareEODPrices.S_DQ_OPEN.unstack(),
            'high':      AShareEODPrices.S_DQ_HIGH.unstack(),
            'low':       AShareEODPrices.S_DQ_LOW.unstack(),
            'close':     AShareEODPrices.S_DQ_CLOSE.unstack(),
            'vwap':      AShareEODPrices.S_DQ_AVGPRICE.unstack(),
            'pctchg':    AShareEODPrices.S_DQ_PCTCHANGE.unstack() / 100,
            'volume':    AShareEODPrices.S_DQ_VOLUME.unstack(),
            'amount':    AShareEODPrices.S_DQ_AMOUNT.unstack(),
            'adjfactor': AShareEODPrices.S_DQ_ADJFACTOR.unstack(),
        }
        path = conf.PATH_DATA_BASIC_PROCESSED / 'stock_quote.pkl'
        if path.exists():
            old = pd.read_pickle(path)
            stock_quote = {k: self.aligner.append(old[k], format_unstack_table(v)) for k, v in stock_quote.items()}
        else:
            stock_quote = {k: self.aligner.align(format_unstack_table(v)) for k, v in stock_quote.items()}
        with open(path, 'wb') as file:
            pickle.dump(stock_quote, file)
    
    def update_stock_size(self, init_date: int = None) -> None:
        if init_date is None:
            init_date = self.aligner.trade_dates[-self.roll_back]
        else:
            init_date = self.aligner.trade_dates[bisect_left(self.aligner.trade_dates[:-self.roll_back], init_date)]
        AShareEODDerivativeIndicator = self.dl.load('AShareEODDerivativeIndicator').loc[str(init_date):]
        stock_size = {
            'total_mv': AShareEODDerivativeIndicator.S_VAL_MV.unstack() * 10000,              # 元
            'float_mv': AShareEODDerivativeIndicator.S_DQ_MV.unstack() * 10000,               # 元
            'total_share': AShareEODDerivativeIndicator.TOT_SHR_TODAY.unstack() * 10000,      # 股
            'float_share': AShareEODDerivativeIndicator.FLOAT_A_SHR_TODAY.unstack() * 10000,  # 股
        }
        path = conf.PATH_DATA_BASIC_PROCESSED / 'stock_size.pkl'
        if path.exists():
            old = pd.read_pickle(path)
            stock_size = {k: self.aligner.append(old[k], format_unstack_table(v)) for k, v in stock_size.items()}
        else:
            stock_size = {k: self.aligner.align(format_unstack_table(v)) for k, v in stock_size.items()}
        with open(path, 'wb') as file:
            pickle.dump(stock_size, file)

    def update_index_quote(self, init_date: int = None) -> None:
        if init_date is None:
            init_date = self.aligner.trade_dates[-self.roll_back]
        else:
            init_date = self.aligner.trade_dates[bisect_left(self.aligner.trade_dates[:-self.roll_back], init_date)]
        AIndexEODPrices = self.dl.load('AIndexEODPrices').loc[str(init_date):]
        index_quote = {
            'open':   AIndexEODPrices.S_DQ_OPEN.unstack(),
            'high':   AIndexEODPrices.S_DQ_HIGH.unstack(),
            'low':    AIndexEODPrices.S_DQ_LOW.unstack(),
            'close':  AIndexEODPrices.S_DQ_CLOSE.unstack(),
            'pctchg': AIndexEODPrices.S_DQ_PCTCHANGE.unstack() / 100,
            'volume': AIndexEODPrices.S_DQ_VOLUME.unstack(),
            'amount': AIndexEODPrices.S_DQ_AMOUNT.unstack(),
        }
        path = conf.PATH_DATA_BASIC_PROCESSED / 'index_quote.pkl'
        for k, v in index_quote.items():
            v.index = v.index.astype(int)
            v = v.reindex(index=self.aligner.trade_dates)
            index_quote[k] = v
        if path.exists():
            old = pd.read_pickle(path)
            for k, v in index_quote.items():
                o = old[k]
                o = o[o.index < v.index.min()]
                v = pd.concat([o, v], axis=0)
                index_quote[k] = v
        with open(path, 'wb') as file:
            pickle.dump(index_quote, file)

    def update_risk_model(self) -> None:
        factor_exposure = pd.read_pickle(conf.PATH_RISK_MODEL_DATA / 'model/factor_exposure.pkl')
        path = conf.PATH_DATA_BASIC / 'FactorExposure.pkl'
        with open(path, 'wb') as file:
            pickle.dump(factor_exposure, file)


if __name__ == "__main__":
    dm = DataMaintainer()
    dm.update_index()
    dm.update_stock_description()
    dm.update_stock_quote()
    dm.update_stock_size()
    dm.update_index_quote()