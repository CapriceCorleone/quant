'''
Author: WangXiang
Date: 2024-04-14 03:06:25
LastEditTime: 2024-04-22 21:40:20
'''

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from typing import Any

from . import formulas
from .moduletools import ModuleManager
from .. import conf
from ..core import DataLoader, Aligner, format_unstack_table
from ..conf.variables import DEVICE_CUPY
from ..core.loader import MinuteBarLoader


def has_cupy():
    try:
        import cupy as cp
        has = True
    except ModuleNotFoundError:
        has = False
    return has


if has_cupy():
    import cupy as cp
    device = cp.cuda.Device(DEVICE_CUPY)
    device.use()
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()


def validate_bar(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.reset_index()
    
    # convert ticker code to int and sort
    index_col = ['date', 'time', 'ticker']
    df['wind_code'] = df['wind_code'].str[:6].astype(int)
    df = df.rename(columns = {'wind_code': 'ticker'})
    df = df.sort_values(['time', 'ticker'])
    
    # drop bad samples
    ticker_count = df['ticker'].value_counts()
    ticker_invalid = ticker_count.loc[ticker_count != ticker_count.max()].index.values
    df = df.loc[~df.ticker.isin(ticker_invalid)]
    df = df.reset_index(drop=True)
    
    # convert dtype to category and float32
    df[index_col] = df[index_col].astype(int).astype('category')
    df[df.columns.difference(index_col)] = df[df.columns.difference(index_col)].astype(np.float32)

    return df


class MinuteBar:

    index_fields = ['date', 'time', 'ticker']

    def __init__(self, quote: pd.DataFrame, cupy: int = False) -> None:
        self.cupy = cupy
        self.fields = quote.columns.difference(self.index_fields)
        self.date = quote.date[0]
        self.tickers = quote.ticker.values.categories.values
        self.time = quote.time.values.categories.values
        self.data = quote[self.fields].values.reshape(240, -1, len(self.fields))
        if self.cupy:
            self.data = cp.array(self.data)
        self.storage = {name: self.data[:, :, i] for i, name in enumerate(self.fields)}
    
    def __len__(self):
        return len(self.tickers)
    
    @property
    def shape(self):
        return self.data.shape
    
    def __getattr__(self, key):
        """
        获取属性的特殊方法。如果属性存在于存储中，则返回存储的值；否则，如果属性为'preclose_2'，则计算并返回值。
        如果属性既不在存储中，也不是'preclose_2'，则跑出AttributeError异常。
        """
        if key in self.storage:
            return self.storage[key].copy()
        elif key == 'preclose_2':
            opens = self.open
            closes = self.close
            value = np.concatenate((opens[:1, :], closes[:-1, :]), axis=0)
            self.storage[key] = value
        elif key == 'ret':
            value = self.close / self.preclose_2 - 1
            self.storage[key] = value
        else:
            raise AttributeError(f"'Quote' object has no attriute '{key}'.")
        return self.__getattr__(key)
    
    def __repr__(self) -> str:
        ticker_summary = f"Tickers ({len(self.tickers)} total):\n"
        ticker_summary += "\t" + ", ".join(self.tickers[:3].astype(str)) + ", ..., " + ", ".join(self.tickers[-3:].astype(str))
        time_summary = f"Times ({len(self.time)} total):\n"
        time_summary += "\t" + ", ".join(self.time[:3].astype(str)) + ", ..., " + ", ".join(self.time[-3].astype(str))
        storage_summary = "Data summary:\n"
        for key, data in self.storage.items():
            storage_summary += f"\tfield: {key}, Shape: {data.shape}\n"
        return f"Date: {self.date}\n\n{time_summary}\n\n{ticker_summary}\n\n{storage_summary}"
    
    def to_array(self, factor):
        dt = np.ones_like(self.tickers) * self.date
        factor = np.stack((dt, self.tickers, factor))
        return factor.T
    
    def add_data(self, name: str, data: Any):
        assert name not in self.storage
        if isinstance(data, pd.Series):
            data = data.loc[self.tickers].values
            if self.cupy:
                data = cp.array(data)
        elif isinstance(data, pd.DataFrame):
            assert len(data) == len(self.time)
            data = data[self.tickers].values
            if self.cupy:
                data = cp.array(data)
        elif not np.isscalar(data):
            raise ValueError(f"data must be a pd.Series or pd.DataFrame or scalar, provided is {type(data)}.")
        self.storage[name] = data
    

class MinuteBarFeatureManager:

    roll_back = 20

    def __init__(self, feature_list, frequency, num_processes = 4, formulas = [formulas], feature_dir = conf.path.PATH_DATA_MINUTE_FEATURE, cupy = True) -> None:
        self.feature_list = feature_list
        self.feature_dir = feature_dir
        self.aligner = Aligner()
        self.frequency = frequency
        self.dl = DataLoader()
        self.trade_dates = self.aligner.trade_dates
        self.bar_loader = MinuteBarLoader(frequency)
        self.num_processes = num_processes
        self.formulas = ModuleManager(formulas)()
        self.cupy = cupy
    
    def calc_feature_at_date(self, date, variables):
        quote = MinuteBar(self.bar_loader[date])
        quote = quote.add_data('limit', variables['limit'])
        quote = quote.add_data('stopping', variables['stopping'])
        result = {}
        for feature in self.feature_list:
            function = self.formulas[feature]
            feature_data = function(quote)
            if isinstance(feature_data, dict):
                if self.cupy:
                    feature_data = {k: quote.to_array(cp.asnumpy(v.squeeze())) for k, v in feature_data.items()}
                else:
                    feature_data = {k: quote.to_array(v.squeeze()) for k, v in feature_data.items()}
            else:
                if self.cupy:
                    feature_data = quote.to_array(cp.asnumpy(feature_data.squeeze()))
                else:
                    feature_data = quote.to_array(feature_data.squeeze())
            result[feature] = feature_data
        del quote
        if self.cupy:
            self.free_cupy_memory()
        return result
    
    @staticmethod
    def free_cupy_memory():
        mempool.free_all_blocks()  # 释放CuPy内存池中的所有块
        pinned_mempool.free_all_blocks()  # 释放CuPy固定内存池中的所有块
    
    def calc_features(self, init_date):
        dl = DataLoader()
        AShareEODPrices = dl.load('AShareEODPrices')
        limit = format_unstack_table(AShareEODPrices.S_DQ_LIMIT.unstack())
        stopping = format_unstack_table(AShareEODPrices.S_DQ_STOPPING.unstack())

        tmp_trade_dates = self.trade_dates[self.trade_dates >= init_date]
        if len(tmp_trade_dates) == 0:
            return
        start_index = self.trade_dates.tolist().index(tmp_trade_dates[0])
        trade_days = self.trade_dates[max(0, start_index - self.roll_back):]
        trade_days = trade_days[trade_days <= max(self.bar_loader.trade_dates)]

        results = []
        pbar = tqdm(total = len(trade_days), ncols = 120, desc = f"Calculating minute-bar features with {self.num_processes} processes(s)")
        if self.num_processes > 1:
            pool = mp.Pool(processes=self.num_processes)
            for dt in trade_days:
                variables = {'limit': limit.loc[dt], 'stopping': stopping.loc[dt]}
                results.append(pool.apply_async(self.calc_feature_at_date, (dt, variables), callback=lambda *args: pbar.update()))
            pool.close()
            pool.join()
            results = [result.get('result') for result in results]
        else:
            for dt in trade_days:
                results.append(self.calc_feature_at_date(dt))
                pbar.update()
        
        hf_features = {feature: [] for feature in self.feature_list}
        for result in results:
            for feature in self.feature_list:
                res = result[feature]
                if isinstance(res, dict):
                    if hf_features[feature] == []:
                        hf_features[feature] = {f: [] for f in res.keys()}
                    for f in res.keys():
                        hf_features[feature][f].append(res[f])
                else:
                    hf_features[feature].append(res)
        return hf_features
    
    def process_features(self, hf_features):
        for feature, data in tqdm(hf_features.items(), desc = 'Saving features', ncols = 80):
            if isinstance(data, list):
                self.__save_feature__(feature, self.to_frame(data))
            else:
                for k, v in data.items():
                    self.__save_feature__(k, self.to_frame(v))
    
    def __save_feature__(self, name, data):
        os.makedirs(self.feature_dir, exist_ok=True)
        save_path = Path(self.feature_dir, name + '.pkl')
        if save_path.exists():
            saved_data = pd.read_pickle(save_path)
            data = self.aligner.append(saved_data, data)
        else:
            data = self.aligner.align(data)
        data.astype(np.float32).to_pickle(save_path)
    
    def to_frame(self, data):
        data = pd.DataFrame(np.concatenate(data), columns=['trade_date', 'ticker', 'factor'])
        data[['trade_date', 'ticker']] = data[['trade_date', 'ticker']].astype(int)
        data = data.set_index(['trade_date', 'ticker'])['factor'].unstack()
        data[np.isinf(data)] = 0
        return data
    
    def run(self, init_date):
        hf_features = self.calc_features(init_date)
        self.process_features(hf_features)


if __name__ == "__main__":
    mbf = MinuteBarFeatureManager(['r_vol', 'r_skew'], '1m')
    hf_features = mbf.run(20240131)