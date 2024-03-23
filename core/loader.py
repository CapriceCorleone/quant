'''
Author: WangXiang
Date: 2024-03-20 22:38:17
LastEditTime: 2024-03-23 14:53:05
'''

import os
import numpy as np
import pandas as pd
from typing import Union, List, Optional

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
    
    def load(self, table: str, suffix: str = 'pkl'):
        if table in self.storage:
            return self.storage[table]
        file_name = table + '.' + suffix
        for dir_path, dir_names, file_names in os.walk(self.root):
            if file_name in file_names:
                path = Path(dir_path, file_name)
                data = self._read_func(suffix)(path)
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


if __name__ == "__main__":
    dl = DataLoader()
    AShareCalendar = dl.load('AShareCalendar')