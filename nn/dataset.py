'''
Author: WangXiang
Date: 2024-04-19 21:28:05
LastEditTime: 2024-04-20 10:23:30
'''

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from ..core import DailyFeatureLoader, Aligner


class DataEngine:

    def __init__(self, device='cpu', **kwargs) -> None:
        """
        Initialize the DataEngine object.

        Args:
            device (str): Specify the device, default is 'cpu'.
            **kwargs: Keyword arguments for registering data groups.
        
        Example:
            label = {'path': PATH_DATA_LABEL, 'type': 'label', 'list': 'simple_label'}

            daily_bar = {
                'path': '../data/feature/daily_bar_feature/',
                'type': 'feature',
                'list': ['close', 'volume']
            }

            style = {
                'path': '../data/feature/style_factor/',
                'type': 'risk',
                'list': None
            }
            data_engine = DataEngine(simple_label = label, daily_bar = daily_bar, style = style)
        """
        aligner = Aligner()
        self.trade_dates = aligner.trade_dates
        self.tickers = aligner.tickers
        self.__is_prepared = False
        self.device = device
        self.meta = pd.DataFrame(index = [], columns = ['data', 'type', 'group', 'path'])

        for group, setting in kwargs.items():
            self.register_data(group, **setting)

    def register_data(self, group, path, type = None, list = None):
        """
        Register a data group.

        Args:
            group (str): Data group name.
            path (str): Data path.
            type (str): Data type, default is 'feature'
            list (list or str): List of data or a single data name, default is None to include all data in the path.
        """
        if self.__is_prepared is False:
            if type is None:
                type = 'feature'
            if list is None:
                list = [os.path.basename(fname).split('.')[0] for fname in os.listdir(path) if fname.endswith('.pkl')]
            if isinstance(list, str): list = [list]
            for data_name in list:
                file_name = data_name + '.pkl'
                data_meta = {
                    'data': data_name,
                    'type': type,
                    'path': os.path.join(path, file_name),
                    'group': group
                }
                if file_name not in os.listdir(path):
                    raise FileExistsError(f"Cannot find {data_name} in {path}.")
                else:
                    self.meta.loc[len(self.meta)] = data_meta
        else:
            raise Exception("Data already prepared, cannot register new.")
    
    def info(self, **kwargs):
        """
        Filter metadata based on specified field values.

        Args:
            **kwargs: Field names and their corresponding values.

        Returns:
            pd.DataFrame: Filtered metadata.
        
        Raises:
            ValueError: If the field is not a valid argument.
        """
        info = self.meta
        for field, value in kwargs.items():
            if field in self.meta.columns:
                if isinstance(value, str):
                    value = [value]
                info = info.loc[info[field].isin(value)]
            else:
                raise ValueError(f"{field} is not valid argument: {self.meta.columns.values.tolist()}.")
        return info.copy()
    
    def index(self, **kwargs):
        """
        Get data indices based on specificed field values.

        Args:
            **kwargs: Field names and their corresponding values.

        Returns:
            dict: Data names and their indices.
        
        Raises:
            Exception: If data is not prepared.
        """
        if self.__is_prepared is True:
            data_list = self.info(**kwargs).data
            return dict(zip(data_list, [self.data_indices[f] for f in data_list]))
        else:
            raise Exception("Data not prepared.")
        
    def prepare(self):
        """
        Prepare the data.
        """
        loader = DailyFeatureLoader(multithreading=True)
        index, self.values = loader.load(self.meta.path.tolist(), stack=True)
        feature_list = index['data']
        self.values = np.r_[
            self.values,
            np.arange(len(self.trade_dates))[None, :, None].repeat(len(self.tickers), axis=2),
            np.arange(len(self.tickers))[None, None, :].repeat(len(self.trade_dates), axis=1)
        ]
        self.values = torch.as_tensor(self.values, dtype=torch.float32, device=self.device)
        self.data_indices = dict(zip(feature_list, range(len(feature_list))))
        self.__is_prepared = True

    @property
    def shape(self):
        return self.values.shape
    
    @property
    def data_names(self):
        return self.meta['data'].values
    
    @property
    def types(self):
        return self.meta['type'].drop_duplicates().values
    
    @property
    def groups(self):
        return self.meta['group'].drop_duplicates().values
    
    @property
    def paths(self):
        return self.meta['path'].values
    
    def to(self, device):
        self.values = self.values.to(device)
        self.device = device
        return self
    
    def __str__(self) -> str:
        desc_info = ''
        for type in self.meta.type.unique():
            m = self.meta.loc[self.meta.type == type]
            desc_info += f'Type: {type}\n'
            for group in m.group.unique():
                g = m.loc[m.group == group]
                folder = os.path.dirname(g.path.iloc[0])
                desc_info += f'  Group: {group}\n'
                desc_info += f'\t-Folder: {folder}\n'
                data_string = f'\t=Data: {", ".join(g.data)}\n'
                # 检查data_string的长度是否超过一行
                if len(data_string) > 80:
                    data_list = g.data
                    data_string = '\t-Data: '
                    line = ''
                    for i, data in enumerate(data_list):
                        if i == 0:
                            line += data
                        elif len(line) + len(data) + 2 > 75:
                            data_string += line + '\n\t\t    '
                            line = data
                        else:
                            line += f', {data}'
                    if line:
                        data_string += line
                    data_string += '\n'
                desc_info += data_string
            desc_info += '\n'
        if not self.__is_prepared:
            return desc_info + '\n' + '!Data not prepared, call .prepare() to load data.'
        else:
            if isinstance(self.values, np.ndarray):
                cpu_mem = self.values.nbytes / (1024 ** 3)
                gpu_mem = 0
            else:
                if self.device == 'cpu':
                    cpu_mem = self.values.element_size() * self.values.nelement() / (1024 ** 3)
                    gpu_mem = 0
                else:
                    cpu_mem = 0
                    gpu_mem = self.values.element_size() * self.values.nelement() / (1024 ** 3)
            desc_status = f"""
            Description:
            Number of data: {len(self.meta)}
            Date range: {self.trade_dates[0]} to {self.trade_dates[1]}, total {len(self.trade_dates)} days
            Types:  {', '.join(self.meta.type.drop_duplicates().tolist())}
            Groups: {', '.join(self.meta.group.drop_duplicates().tolist())}
            Shape:  {self.shape}

            Memory info:
            Dtype: {self.values.dtype}
            Device: {self.device}
            Estimated CPU memory usage: {cpu_mem:.4f}GB
            Estimated GPU memory usage: {gpu_mem:.4f}GB            """

            desc_status = '\n'.join(line.strip() for line in desc_status.split('\n'))

            return desc_info + '\n' + desc_status
    
    __repr__ = __str__


class DefaultDataset(Dataset):

    def __init__(self, data_engine: DataEngine, start_date: int, end_date: int, universe: pd.DataFrame, device = 'cpu', **dataset_args):
        self.data_engine = data_engine
        self.start_date = start_date
        self.end_date = end_date
        self.universe = torch.as_tensor(universe.values, dtype=bool).to(device)
        self.device = device
        self.create_index()
        self.splits_slice = {}
        self.parse_args(dataset_args)

    def create_index(self):
        self.create_date_index()
        self.create_feature_index()

    def create_date_index(self):
        self.date_index = np.where((self.data_engine.trade_dates >= self.start_date) & (self.data_engine.trade_dates <= self.end_date))[0]
    
    def create_feature_index(self):
        self.feature_index = {group: sorted(list(self.data_engine.index(group=group).values())) for group in self.data_engine.groups()}
        self.feature_index_all = sum(list(self.feature_index.values()), [])
    
    def parse_args(self, dataset_args):
        self.dataset_args = dataset_args
        self.window = dataset_args.get('window', 1)
        self.patch = dataset_args.get('patch', 1)
        self.splits = dataset_args.get('splits',
                                       {
                                           'inputs': {'type': ['feature']},
                                           'labels': {'type': ['label']}
                                       })
        for key, split in self.splits.items():
            if isinstance(split, dict):
                slice_ = self.get_slice(**split)
            else:
                slice_ = sorted(list(set(sum([self.get_slice(**i) for i in split], []))))
            self.splits_slice[key] = torch.LongTensor(slice_).to(self.device)
    
    def get_slice(self, **kwargs):
        out = self.data_engine.index(**kwargs)
        return sorted(list(out.values()))

    def batching(self, i: int) -> torch.Tensor:
        data = self.data_engine.values[:, i - self.window * self.patch + self.patch : i + self.patch : self.patch].permute(1, 2, 0).to(self.device)
        universe = self.universe[i]
        return data[:, universe, :]
    
    def __len__(self):
        return len(self.date_index)
    
    def __iter__(self):
        return iter(self.date_index)
    
    def __getitem__(self, i):
        return self.batching(self.date_index[i])