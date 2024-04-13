'''
Author: WangXiang
Date: 2024-03-23 21:32:14
LastEditTime: 2024-03-30 15:41:42
'''

import os
import yaml
import pickle
import inspect
import numpy as np
import pandas as pd
from datetime import datetime

from .. import conf
from ..core import DataLoader, Aligner, Universe, format_unstack_table


class RiskModelManager:

    root = conf.PATH_RISK_MODEL_DATA

    def __init__(self, structure_file: str, init_date: int = 20081231, num_process: int = 1) -> None:
        with open(structure_file, 'r', encoding='utf-8') as file:
            self.structure = yaml.load(file, yaml.FullLoader)
        self.dl = DataLoader()
        self.aligner = Aligner()
        self.init_date = init_date
        self.num_process = num_process
        self.univ = Universe()
        self.init_universe = self.univ()
        self.universe = self.univ(listed_days=122, include_st=False)
        self.tech_universe = self.univ(listed_days=63)
        self.weight = self.dl.load('stock_size')['total_mv'] / 1e8
    
    def calc_factor(self, config, **kwargs):
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
        if 'kwargs' in config:
            kwargs.update(config['kwargs'])
        factor = func(**args, **kwargs)
        factor = format_unstack_table(factor)
        
        # correct dates
        factor_date_diff = np.setdiff1d(factor.index.values, self.aligner.trade_dates)
        if len(factor_date_diff) > 0:
            factor_date_replace_diff = {d: self.aligner.trade_dates[self.aligner.trade_dates <= d].max() for d in factor_date_diff}
            factor.index = list(map(lambda x: factor_date_replace_diff.get(x, x), factor.index))
            factor.index.name = 'trade_date'
            factor = factor.reset_index().drop_duplicates('trade_date', keep='last').set_index('trade_date')
        
        factor = self.aligner.align(factor)
        factor[self.init_universe == 0] = np.nan
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
    
    def calc_risk_subfactor(self, config):
        kwargs = {}
        if config['universe'] is not None:
            kwargs['universe'] = getattr(self, config['universe'])
        if config['weight'] is not None:
            kwargs['weight'] = self.weight
        factor = self.calc_factor(config, **kwargs)
        if 'process' in config:
            factor = self.process_factor(factor, config['process'])
        return factor
    
    def calc_risk_factor(self, config):
        subfactors = {}
        for i in range(len(config['subfactors'])):
            subfactors[config['subfactors'][i]['name']] = self.calc_risk_subfactor(config['subfactors'][i])
        factor = pd.DataFrame(np.nanmean(list(subfactors.values()), axis=0), index=self.aligner.trade_dates, columns=self.aligner.tickers)
        if 'process' in config:
            factor = self.process_factor(factor, config['process'])
        return factor, subfactors
    
    def calc_exposure(self):
        factors = {}
        subfactors = {}
        for config in self.structure:
            print(config['name'])
            if config['type'] == 'market':
                factor = pd.DataFrame(1, index=self.aligner.trade_dates, columns=self.aligner.tickers)
                factors[config['name']] = factor
            elif config['type'] == 'style':
                factor, subfactor = self.calc_risk_factor(config)
                factors[config['name']] = factor
                subfactors.update(subfactor)
            elif config['type'] == 'industry':
                AShareIndustriesClassCITICS = self.dl.load('AShareIndustriesClassCITICS')
                for name in config['name']:
                    info = AShareIndustriesClassCITICS[AShareIndustriesClassCITICS['INDUSTRIESNAME'] == name][['S_INFO_WINDCODE', 'ENTRY_DT', 'REMOVE_DT']]
                    factor = self.univ._format_universe(self.univ.arrange_info_table(info))
                    factors[name] = factor
            else:
                raise ValueError (f"Invalid factor configuration for risk model: {config}.")
        factors = {k: v.loc[self.init_date:] for k, v in factors.items()}
        subfactors = {k: v.loc[self.init_date:] for k, v in subfactors.items()}
        for name, data in factors.items():
            self.save_factor(data, self.root / f'factor/{name}.pkl')
        for name, data in subfactors.items():
            self.save_factor(data, self.root / f'subfactor/{name}.pkl')
        self.save_factors(factors, self.root / 'model/factor_exposure.pkl')
        return factors, subfactors
    
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

    def save_factors(self, data, path):
        print(f'{datetime.now()}: Saving factor to {path}')
        os.makedirs(path.parent, exist_ok=True)
        if path.exists():
            old = pd.read_pickle(path)
            data = {k: self.aligner.append(old[k], v) for k, v in data.items()}
        else:
            data = {k: self.aligner.align(v) for k, v in data.items()}
        with open(path, 'wb') as file:
            pickle.dump(data, file)
        print(f'\tLast date = {data[list(data.keys())[0]].index.max()}')

    def save_data(self, data, path):
        print(f'{datetime.now()}: Saving data to {path}')
        os.makedirs(path.parent, exist_ok=True)
        if path.exists():
            old = pd.read_pickle(path)
            old = old[old['trade_date'] < data['trade_date'].min()]
            data = pd.concat([old, data], axis=0, ignore_index=True)
        data.to_pickle(path)
        print(f"\tLast date = {data['trade_date'].max()}")