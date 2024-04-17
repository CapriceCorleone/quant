'''
Author: WangXiang
Date: 2024-03-20 22:38:17
LastEditTime: 2024-04-14 02:34:11
'''

import os
import numpy as np
import pandas as pd
import concurrent.futures
from typing import Union, List, Optional, Any

from .. import conf
from pathlib import Path


class DataLoader:
    """
    数据载入模块
    通过DataLoader.help来查询本地数据
    """
    root = conf.path.PATH_DATA

    def __init__(self, save: bool = True) -> None:
        """
        :param save: 是否将数据存入内存，以便下次直接从内存获取
        """
        self.save = save
        self.storage = {}
    
    def load(self, table: str, suffix: str = 'pkl', keys: List[str] = None):
        if table in self.storage:
            return self.storage[table]
        file_name = table + '.' + suffix
        for dir_path, dir_names, file_names in os.walk(self.root):
            if file_name in file_names:
                path = Path(dir_path, file_name)
                data = self._read_func(suffix)(path)
                if keys is not None:
                    data = {k: v for k, v in data.items() if k in keys}
                if self.save:
                    self.storage[table] = data
                return data
        raise ValueError (f"Cannot find file {table} in {self.root}.")

    def _read_func(self, suffix: str = 'pkl'):
        if suffix == 'pkl':
            return pd.read_pickle
        if suffix == 'xlsx':
            return pd.read_excel
        if suffix == 'csv':
            return pd.read_csv
        if suffix == '.parquet':
            return pd.read_parquet
        raise ValueError (f"Cannot read {suffix} files.")

    def help(self):
        """
        打印文件列表
        """
        for dir_path, dir_names, file_names in os.walk(self.root):
            if len(file_names) > 0:
                print(f'{dir_path}:')
                if 'Quote' not in dir_path or 'Book' not in dir_path:
                    for n in file_names:
                        print('\t' + n)


class MinuteBarLoader:

    def __init__(self, frequency: Any = '1m', path: str = None, multithreading: bool = True) -> None:
        if path is None:
            self.path = conf.path.PATH_DATA_MINUTE_BAR[frequency]
        else:
            self.path = path
        self.files = os.listdir(self.path)
        self.trade_dates = np.array([int(fname.split('.')[0]) for fname in self.files])
        self.use_multithread = multithreading

    def __getitem__(self, date: Union[int, slice]):
        if np.issubdtype(type(date), np.integer):
            if date not in self.trade_dates:
                raise ValueError (f"Cannot find data at {date} in {self.path}.")
            return pd.read_pickle(self.path / f'{date}.pkl')
        if isinstance(date, slice):
            date_list = self.trade_dates[(self.trade_dates >= date.start) & (self.trade_dates <= date.stop)]
            if self.use_multithread:
                return self._multithread_load(date_list)
            return [self.__getitem__(int(date)) for date in date_list]
    
    def _multithread_load(self, date_list):
        result = [None] * len(date_list)
        progress = [0] * len(date_list)
        def load_to_result(i):
            progress[i] = 1
            result[i] = self.__getitem__(int(date_list[i]))
            print(f"\rLoading minute bars from {date_list[0]} to {date_list[-1]} ..."
                  f"{100 * sum(progress) / len(date_list):.2f}%", end = "")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(load_to_result, range(len(date_list)))
        return result


class DailyFeatureLoader:

    def __init__(self, multithreading: bool = True) -> None:
        self.multithreading = multithreading
    
    def load_single_file(self, file: str, return_value = False) -> pd.DataFrame:
        data = pd.read_pickle(file)
        if return_value:
            data = data.values
        return data
    
    def load(self, f: Union[str, List], feature_list: List = None, stack: bool = False):
        """
        Load data from a single file or a list of files.
        Args
            f: Union[str, List]: data folder path / feature abs paths / list of feature abspath
            feature_list: List, optional: list of feature name, by default None
            stack: bool, optional: whether to stack features into a 3-D array
        Returns:
            if stack = False:
                Tuple[List, List]: List of data names, List of loaded features
            if stack = True:
                Tuple[Dict, np.ndarray]: Dictionary with index, columns and data names, 3-D array of stacked features
        """
        if isinstance(f, str):
            if os.path.isdir(f):
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Cannot find path: {f}.")
                file_in_f = os.listdir(f)
                if feature_list is None:
                    feature_list = [file.split('.')[0] for file in file_in_f]
                invalid_file = [feat for feat in feature_list if feat + '.pkl' not in file_in_f]
                if len(invalid_file) > 0:
                    file_str = ','.join(invalid_file)
                    raise FileNotFoundError(f"Cannot find file: {file_str} in path: {f}.")
                feat_paths = [os.path.join(f, feat + '.pkl') for feat in feature_list]
            elif os.path.isfile(f):
                feature_list = f
                result = self.load_single_file(f)
                return feature_list, result
            else:
                raise FileNotFoundError(f"Cannot find path: {f}.")
        elif isinstance(f, list):
            invalid_file = list(filter(lambda x: not os.path.exists(x), f))
            if len(invalid_file) > 0:
                file_str = ','.join(invalid_file)
                raise FileNotFoundError(f"Cannot find file: {file_str}.")
            feature_list = [os.path.basename(fname).split('.')[0] for fname in f]
            feat_paths = f
        else:
            raise TypeError(f"Unsupported type: {type(f)}.")
        
        if stack:
            first_df = self.load_single_file(feat_paths[0])
            container = np.empty((len(feat_paths), first_df.shape[0], first_df.shape[1]), dtype=np.float32)
            index = {'index': first_df.index.values, 'columns': first_df.columns.values}
        else:
            container = [None] * len(feat_paths)
        
        if self.multithreading:
            container = self._multithread_load(feat_paths, container, stack)
        else:
            for i in range(len(feat_paths)):
                container[i] = self.load_single_file(feat_paths[i], stack)
        
        if stack:
            index['data'] = feature_list
            return index, container
        else:
            return feature_list, container
        
    def _multithread_load(self, feat_paths: List[str], container, return_value: bool = False):
        """
        Multithreaded loading of files into a container.
        Args:
            feat_paths (List[str]): List of feature paths to load.
            container: The container to load the files into.
            return_value (bool, optional): Whether to return the loaded container. Defaults to False.
        Returns:
            The container with the loaded files.
        """
        def load_to_result(i):
            data = self.load_single_file(feat_paths[i], return_value)
            if isinstance(container, np.ndarray):
                if data.shape[0] != container.shape[1] or data.shape[1] != container.shape[2]:
                    raise ValueError(f"Shape mismatch: {data.shape} vs {container.shape}.")
            container[i] = self.load_single_file(feat_paths[i], return_value)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(load_to_result, range(len(feat_paths)))
        return container


if __name__ == "__main__":
    dl = DataLoader()
    AShareCalendar = dl.load('AShareCalendar')