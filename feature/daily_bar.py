'''
Author: WangXiang
Date: 2024-04-14 02:39:18
LastEditTime: 2024-04-14 03:31:46
'''

import os
import inspect
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from . import formulas
from .moduletools import ModuleManager
from .. import conf
from ..core import DataLoader, Aligner


class DailyBarFeatureManager:

    roll_back = 20

    def __init__(self, feature_list, formulas = [formulas], feature_dir = conf.path.PATH_DATA_DAILY_FEATURE) -> None:
        self.feature_list = feature_list
        self.feature_dir = feature_dir
        self.dl = DataLoader()
        self.aligner = Aligner()
        self.trade_dates = self.aligner.trade_dates
        self.formulas = ModuleManager(formulas)()
    
    def free_dl_memory(self):
        self.dl.storage = {}
    
    def calc_features(self, init_date):
        tmp_trade_dates = self.trade_dates[self.trade_dates >= init_date]
        if len(tmp_trade_dates) == 0:
            return
        start_index = self.trade_dates.tolist().index(tmp_trade_dates[0])
        trade_days = self.trade_dates[max(0, start_index - self.roll_back):]
        start_date = trade_days[0]
        results = {}
        for feature in self.feature_list:
            function = self.formulas[feature]
            fullargspec = inspect.getfullargspec(function)
            args = [self.dl.load(arg) for arg in fullargspec.args if arg != 'init_date']
            feature_data = function(*args, init_date=start_date)
            if isinstance(feature_data, dict):
                feature_data = {k: v for k, v in feature_data.items()}
            results[feature_data] = feature_data
        self.free_dl_memory()
        return results
    
    def format_features(self, features):
        outputs = {}
        for feature, data in features.items():
            if isinstance(data, dict):
                for k, v in data.items():
                    outputs[k] = v
            else:
                outputs[feature] = data
        return outputs
    
    def save_features(self, features):
        os.makedirs(self.feature_dir, exist_ok=True)
        for feature, data in features.items():
            save_path = Path(self.feature_dir, feature + '.pkl')
            if save_path.exists():
                saved_data = pd.read_pickle(save_path)
                data = self.aligner.append(saved_data, data)
            else:
                data = self.aligner.align(data)
            data.astype(np.float32).to_pickle(save_path)
    
    def run(self, init_date):
        features = self.calc_features(init_date)
        features = self.format_features(features)
        self.save_features(features)
        return features