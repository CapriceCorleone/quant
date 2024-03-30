'''
Author: WangXiang
Date: 2024-03-30 15:21:58
LastEditTime: 2024-03-30 15:47:47
'''

import os
import yaml
import inspect
import traceback
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime

from .. import conf
from ..core import DataLoader, Aligner, Universe, format_unstack_table

class FactorManager:

    root = conf.PATH_FACTOR_LAB
    
    def __init__(self, config_file: str, init_date: int = 20081231, num_process: int = 1) -> None:
        with open(config_file, 'r', encoding='utf-8') as file:
            self.config = yaml.load(file, yaml.FullLoader)
        self.dl = DataLoader()
        self.aligner = Aligner()
        self.init_date = init_date
        self.num_process = num_process
        self.univ = Universe()
        self.init_universe = self.univ()
        self.data_folder = self.root / 'data'
        self.primitive_folder = self.data_folder / 'primitive'
        self.standardized_folder = self.data_folder / 'standardized'
        self.orthogonalized_folder = self.data_folder / 'orthogonalized'
    
    def calc_factor(self, config):
        func = config['formulas']
        arg_names = inspect.getfullargspec(func).args
        args = {}
        for name in arg_names:
            if name == 'init_date':
                args[name] = self.init_date
            elif name == 'num_process':
                args[name] = self.num_process
            else:
                args[name] = self.dl.load(name)
        factor = func(**args)
        factor = format_unstack_table(factor)
        factor = self.aligner.align(factor)
        factor[self.init_universe] = np.nan
        return factor
    
    def process_factor(self, factor, processes):
        for func in processes:
            arg_names = inspect.getfullargspec(func).args
            args = {}
            for name in arg_names:
                if name == 'weight':
                    args['weight'] = self.weight
                elif name == 'data':
                    args['data'] = factor
                else:
                    raise ValueError (f"Invalid args for factor processing: {name}.")
            factor = func(**args)
        return factor
    
    def save_factor(self, data, path):
        print(f'{datetime.now()}: Saving factor to {path}')
        os.makedirs(path.parent, exist_ok=True)
        if path.exists():
            old = pd.read_pickle(path)
            data = self.aligner.append(old, data)
        else:
            data = self.aligner.align(data)
        data.to_pickle(path)
        print(f'\tLast date = {data.index.max()}')
    
    def run(self, job_list: List[str] = None) -> None:
        for factor_name, factor_config in self.config.items():
            if job_list is None:
                execute = True
            else:
                execute = factor_name in job_list
            if execute:
                try:
                    factor = self.calc_factor(factor_config)
                    primitive_path = self.primitive_folder / f'{factor_name}.pkl'
                    self.save_factor(factor, primitive_path)
                    if 'standardize' in factor_config:
                        factor = self.process_factor(factor, factor_config['standardize'])
                    standardized_path = self.standardized_folder / f'{factor_name}.pkl'
                    self.save_factor(factor, standardized_path)
                    if 'orthogonalize' in factor_config:
                        factor = self.process_factor(factor, factor_config['orthogonalize'])
                    orthogonalized_path = self.orthogonalized_folder / f'{factor_name}.pkl'
                    self.save_factor(factor, orthogonalized_path)
                except:
                    print(f'An error has occurred when executing calculation and processes for factor [{factor_name}].')
                    traceback.print_exc()

