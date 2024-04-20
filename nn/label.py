'''
Author: WangXiang
Date: 2024-04-20 14:35:29
LastEditTime: 2024-04-20 15:28:05
'''

import os
import gc
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
warnings.filterwarnings('ignore')

from .. import conf
from ..conf.variables import DEVICE_CUPY
from ..core import DataLoader, Aligner, format_unstack_table


def has_cupy():
    try:
        import cupy as cp
        has = True
    except ModuleNotFoundError:
        has = False
    return has



HAS_CUPY = has_cupy()
if HAS_CUPY:
    import cupy as cp
    device = cp.cuda.Device(DEVICE_CUPY)
    device.use()
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()


class LabelManager:

    def __init__(self, dir_label=conf.PATH_DATA_LABEL) -> None:
        self.aligner = Aligner()
        self.dir_label = dir_label
    
    def calc_label(self, init_date):
        raise NotImplementedError(f"Error: method 'calc_label` is not implemented for {self.__name__}.")
    
    def save_label(self, data: pd.DataFrame):
        os.makedirs(self.dir_labal)
        save_path = Path(self.dir_label, self.label_name + '.pkl')
        if save_path.exists():
            saved_data = pd.read_pickle(save_path)
            data = self.aligner.append(saved_data, data)
        else:
            data = self.aligner.align(data)
        data.to_pickle(save_path)

    def run(self):
        raise NotImplementedError(f"Error: method 'run' is not implemented for {self.__name__}.")
    
    def __call__(self, init_date: int):
        return self.run(init_date)
    

class SimpleLabelManager(LabelManager):
    
    def __init__(self, window: int, skip: int = 1, buy: str = 'close', label_name: str = 'simple', dir_label=conf.PATH_DATA_LABEL) -> None:
        super().__init__(dir_label)
        dl = DataLoader()
        AShareEODPrices: pd.DataFrame = dl.load('AShareEODPrices')
        adjclose = (AShareEODPrices.S_DQ_CLOSE * AShareEODPrices.S_DQ_ADJFACTOR).unstack()
        adjclose = format_unstack_table(adjclose)
        self.sell = adjclose
        if buy == 'close':
            self.buy = adjclose
        elif buy == 'vwap':
            adjvwap = (AShareEODPrices.S_DQ_AVGPRICE * AShareEODPrices.S_DQ_ADJFACTOR).unstack()
            adjvwap = format_unstack_table(adjvwap)
            self.buy = adjvwap
        elif buy == 'open':
            adjopen = (AShareEODPrices.S_DQ_OPEN * AShareEODPrices.S_DQ_ADJFACTOR).unstack()
            adjopen = format_unstack_table(adjopen)
            self.buy = adjopen
        else:
            raise ValueError(f"Cannot identify the 'buy' style for '{buy}'.")
        self.window = window
        self.skip = skip
        self.trade_dates = self.buy.index.values

    def calc_label(self, init_date: int) -> pd.DataFrame:
        start_date = self.trade_dates[self.trade_dates <= init_date].max() if init_date >= self.trade_dates[0] else self.trade_dates[0]
        label = self.sell.shift(-self.window - self.skip).loc[start_date:] / self.buy.shift(-self.skip).loc[start_date:] - 1
        return label
    
    def run(self, init_date: int) -> pd.DataFrame:
        label = self.calc_label(init_date)
        self.save_label(label)
        return label


class SimilarLabelManager(SimpleLabelManager):

    def __init__(self, window: int, similar_window: int, skip: int = 1, buy: str = 'close', num_similar: int = 30, label_name: str = 'simple', dir_label=conf.PATH_DATA_LABEL) -> None:
        super().__init__(window, skip, buy, label_name, dir_label)
        self.similar_window = similar_window
        self.num_similar = num_similar
        self.daily_ret = self.sell / self.sell.shift(1) - 1
    
    @staticmethod
    def free_cupy_memory():
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    def calc_raw_label(self) -> pd.DataFrame:
        return self.sell.shift(-self.window - self.skip) / self.buy.shift(-self.skip) - 1
    
    def calc_daily_ret(self) -> pd.DataFrame:
        return self.daily_ret.fillna(value=0)

    def calc_label_at_date(self, i: int, __mod__):
        day = self.trade_dates_cp[i]
        ret = self.daily_ret_cp[i - self.similar_window + 1 : i + 1]
        corr = __mod__.corrcoef(ret.T)
        corr = corr - __mod__.eye(len(corr)) * 3
        corr = np.nan_to_num(corr, nan=-2)
        loc = __mod__.argpartition(corr, -self.num_similar, axis=0)[-self.num_similar:, :]
        label = self.raw_label_cp[i]
        similar_label = self.raw_label_cp[i, __mod__.ravel(loc)].reshape(loc.shape)
        mean = np.nanmean(similar_label, axis=0)
        std = np.nanstd(similar_label, axis=0, ddof=1)
        median = np.nanmedian(similar_label, axis=0)
        mad = np.nanmedian(np.abs(similar_label - median), axis=0)
        max_ = np.nanmax(similar_label, axis=0)
        min_ = np.nanmin(similar_label, axis=0)
        label = label - median
        p = __mod__.c_[__mod__.ones(len(self.tickers_cp)) * day.item(), self.tickers_cp, label, mean, std, median, mad, max_, min_]
        if HAS_CUPY:
            p = cp.asnumpy(p)
        del day, ret, corr, loc, label, similar_label, mean, std, median, mad, max_, min_
        if HAS_CUPY:
            self.free_cupy_memory()
        gc.collect()
        return p
    
    def calc_label(self, init_date: int) -> pd.DataFrame:
        start_date = self.trade_dates[self.trade_dates <= init_date].max() if init_date >= self.trade_dates[0] else self.trade_dates[0]
        start_ix = max(self.trade_dates.tolist().index(start_date), self.similar_window - 1)
        raw_label = self.calc_raw_label()
        daily_ret = self.calc_daily_ret()
        assert (raw_label.index.values == daily_ret.index.values).all()
        assert (raw_label.columns.values == daily_ret.columns.values).all()
        if HAS_CUPY:
            __mod__ = cp
        else:
            __mod__ = np
        self.trade_dates_cp = __mod__.array(raw_label.index.values)
        self.tickers_cp = __mod__.array(raw_label.columns.values)
        self.raw_label_cp = __mod__.array(raw_label.values)
        self.daily_ret_cp = __mod__.array(daily_ret.values)

        results = []
        for i in tqdm(range(start_ix, len(daily_ret)), ncols=80, desc='calc label'):
            p = self.calc_label_at_date(i, __mod__=__mod__)
            results.append(p)

        delattr(self, 'trade_dates_cp')
        delattr(self, 'tickers_cp')
        delattr(self, 'raw_label_cp')
        delattr(self, 'daily_ret_cp')

        results = pd.DataFrame(np.concatenate(results, axis=0), columns=['trade_date', 'ticker', 'labels', 'mean', 'std', 'median', 'mad', 'max', 'min'])
        results[['trade_date', 'ticker']] = results[['trade_date', 'ticker']].astype(int)
        results = results.sort_values(['trade_date', 'ticker'])
        return results.pivot(index='trade_date', columns='ticker', values='labels')
    

