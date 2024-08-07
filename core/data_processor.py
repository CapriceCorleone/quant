'''
Author: WangXiang
Date: 2024-03-24 17:12:39
LastEditTime: 2024-03-24 17:31:13
'''

import os
import inspect
import pandas as pd
from typing import List
from datetime import datetime
from abc import abstractmethod

from .. import conf
from .loader import DataLoader
from .calendar import Calendar


class DataProcessTask:

    def __init__(self) -> None:
        self.task = 'full'
        self.field_date = None
        assert (self.task in ['full', 'incremental'])

    @abstractmethod
    def run(self) -> None:
        pass


class DataProcessor:
    """
    数据处理流程模块
    """
    root = conf.PATH_DATA / 'processed'
    
    def __init__(self, task_list: List, roll_back: int = 20) -> None:
        self.calendar = Calendar()
        self.task_list = task_list
        os.makedirs(self.root, exist_ok=True)
        self.dl = DataLoader()
        self.roll_back = roll_back

    def run(self, init_date: int) -> None:
        for task in self.task_list:
            print(f'{datetime.now()}: Processing table [{task.__name__}]')
            path = self.root / f'{task.__name__}.pkl'
            task_obj = task()
            arg_names = inspect.getfullargspec(task_obj.run).args
            args = {}
            for name in arg_names:
                if name not in ('self', 'init_date'):
                    args[name] = self.dl.load(name)
            assert task_obj.task in ('full', 'incremental')
            if task_obj.task == 'full':
                processed = task_obj.run(**args)
            elif task_obj.task == 'incremental':
                if path.exists():
                    file_existed = pd.read_pickle(path)
                    file_date = file_existed[task_obj.field_date].max()
                    init_date = self.calendar.get_prev_trade_date(init_date, self.roll_back)
                    file_existed = file_existed.loc[file_existed[task_obj.field_date] <= int(init_date)]
                    args['init_date'] = init_date
                    processed = task_obj.run(**args)
                    processed = processed.loc[processed[task_obj.field_date] > init_date]
                    processed = pd.concat([file_existed, processed], axis=0).drop_duplicates().reset_index(drop=True)
                else:
                    init_date = 20050104
                    args['init_date'] = init_date
                    processed = task_obj.run(**args)
            processed.to_pickle(path)


