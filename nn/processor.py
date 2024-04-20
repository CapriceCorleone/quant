'''
Author: WangXiang
Date: 2024-04-19 23:16:18
LastEditTime: 2024-04-20 10:33:30
'''

import gc
import copy
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Mapping

import torch
from torch import Tensor

from .dataset import DefaultDataset


# %% pre process
class ProcessorBase:

    def __init__(self, train=True, pred=True, device='cuda:0') -> None:
        self.train = train
        self.pred = pred
        self.device = device

    def __call__(self):
        raise NotImplementedError(f"__call__ method is not implemented for {self.__name__}")


class LastValueNormalizer(ProcessorBase):
    """
    Input: data (timesteps, tickers, feature) = (L, N, F)
    Function: divide the certain features set (slice) by their last value
    """
    def __init__(self, slice1, slice2=None, train=True, pred=True, device='cuda:0') -> None:
        super().__init__(train, pred, device)
        self.slice1 = torch.LongTensor(slice1).to(device)
        self.slice2 = slice1 if slice2 is None else torch.LongTensor(slice2).to(device)

    def __call__(self, data: Tensor):
        data[:, :, self.slice1] = data[:, :, self.slice1] / data[-1, :, self.slice2]
        return data


class FilterOutValule(ProcessorBase):
    """
    Input: data (timesteps, tickers, feature) = (L, N, F)
    Function: keep the tickers which don't contains any/all element that equals `value` in the certain dates (slice1) and the certain features set (slice2)
    """
    def __init__(self, value, slice1, slice2, how='all', train=True, pred=True, device='cuda:0') -> None:
        super().__init__(train, pred, device)
        self.value = value
        self.slice1 = torch.LongTensor(slice1).to(device)
        self.slice2 = torch.LongTensor(slice2).to(device)
        self.filter_func = torch.all if how == 'any' else torch.any
    
    def __call__(self, data: Tensor):
        return data[:, self.filter_func(self.filter_func(~(data[self.slice1][:, :, self.slice2] == self.value), dim=0), dim=-1), :]


class FilterInAllPositive(ProcessorBase):
    """
    Input: data (timesteps, tickers, feature) = (L, N, F)
    Function: keep the tickers which are all positive in the certain dates (slice1) and the certain features set (slice2)
    """
    def __init__(self, slice1, slice2, train=True, pred=True, device='cuda:0') -> None:
        super().__init__(train, pred, device)
        self.slice1 = torch.LongTensor(slice1).to(device)
        self.slice2 = torch.LongTensor(slice2).to(device)

    def __call__(self, data: Tensor):
        filtered = data[self.slice1][:, :, self.slice2]
        return data[:, filtered.min(dim=0).values.min(dim=-1).values > 0, :]
    

class Clip(ProcessorBase):
    """
    Input:
        data (timesteps, tickers, feature) = (L, N, F)
        min_value: number
        max_value: number
    Function: clip the certain features set (slice) by min and max parameters provided
    """
    def __init__(self, slice, min_value, max_value, train=True, pred=True, device='cuda:0') -> None:
        super().__init__(train, pred, device)
        self.min_value = min_value
        self.max_value = max_value
        self.slice = torch.LongTensor(slice).to(device)
    
    def __call__(self, data: Tensor):
        data[:, :, self.slice] = torch.clip(data.index_select(2, self.slice), self.min_value, self.max_value)
        return data


class FilterInNotAllZero(ProcessorBase):
    """
    Input: data (timesteps, tickers, feature) = (L, N, F)
    Function: keep the tickers which don't have any all-zero feature in the certain dates (slice1) and the certain features set (slice2)
    """
    def __init__(self, slice1, slice2, train=True, pred=True, device='cuda:0') -> None:
        super().__init__(train, pred, device)
        self.slice1 = torch.LongTensor(slice1).to(device)
        self.slice2 = torch.LongTensor(slice2).to(device)
    
    def __call__(self, data: Tensor):
        filtered = data[self.slice1][:, :, self.slice2]
        return data[:, torch.abs(filtered).max(dim=-1).values.min(dim=0).values > 0, :]


class FilterInRange(ProcessorBase):
    """
    Input: data (timesteps, tickers, feature) = (L, N, F)
    Function: keep the tickers which don't contains any/all element that lay out of the boundary (lower ~ upper) in the certain dates (slice1) and the certain features set (slice2)
    """
    def __init__(self, lower, upper, slice1, slice2, how='all', train=True, pred=True, device='cuda:0') -> None:
        super().__init__(train, pred, device)
        self.lower = lower
        self.upper = upper
        self.slice1 = torch.LongTensor(slice1).to(device)
        self.slice2 = torch.LongTensor(slice2).to(device)
        self.filter_func = torch.all if how == 'all' else torch.any
    
    def __call__(self, data: Tensor):
        filtered = data[self.slice1][:, :, self.slice2]
        return data[:, self.filter_func(self.filter_func((filtered >= self.lower) & (filtered <= self.upper), dim=0), dim=-1), :]


class Dropna(ProcessorBase):
    
    def __init__(self, slice1, slice2, how='any', train=True, pred=True, device='cuda:0') -> None:
        super().__init__(train, pred, device)
        self.slice1 = torch.LongTensor(slice1).to(device)
        self.slice2 = torch.LongTensor(slice2).to(device)
        self.filter_func = torch.all if how == 'any' else torch.any

    def __call__(self, data: Tensor):
        return data[:, self.filter_func(self.filter_func(~torch.isnan(data[self.slice1][:, :, self.slice2]), dim=0), dim=-1), :]


class Fillna(ProcessorBase):

    def __init__(self, nan=0, train=True, pred=True, device='cuda:0') -> None:
        super().__init__(train, pred, device)
        self.nan = nan
    
    def __call__(self, data: Tensor):
        return torch.nan_to_num(data, nan=self.nan)


class SelfStandardScaler(ProcessorBase):
    """
    Input:
        data (timesteps, tickers, feature) = (L, N, F)
    Function: scale the certain features set (slice2) for the certain timesteps (slice1) by mean and std parameters of the input data (mean and std are calculated instantaneouly along the first dimension)
    """
    def __init__(self, slice1, slice2, train=True, pred=True, device='cuda:0', eps=1e-8) -> None:
        super().__init__(train, pred, device)
        self.slice1 = torch.LongTensor(slice1).to(device)
        self.slice2 = torch.LongTensor(slice2).to(device)
        self.indices1 = self.slice1.repeat(len(slice2))
        self.indices2 = self.slice2.tile(len(slice1))
        self.eps = eps
    
    def __call__(self, data: Tensor):
        data = data.permute(0, 2, 1)
        slice_data = data[self.indices1, self.indices2]
        N = (~torch.isnan(slice_data)).sum(dim=-1, keepdim=True)
        mean = slice_data.nanmean(dim=-1, keepdim=True)
        std = torch.sqrt(((slice_data ** 2).nanmean(dim=-1, keepdim=True) - mean ** 2) * N / (N - 1))
        data = data.index_put((self.indices1, self.indices2), (slice_data - mean) / (std + 1e-8))
        data = data.permute(0, 2, 1)
        return data


# %% post process (must have `params` and `params_slice`)
class MinMaxClip(ProcessorBase):
    """
    Input:
        data (timesteps, tickers, feature) = (L, N, F)
        min (timesteps, feature) = (L, F)
        max (timesteps, feature) = (L, F)
    Function: clipi the certain features set (slice) by min and max parameters (must be 2-D tensor) provided
    """
    def __init__(self, params: Dict[str, Tensor], slice, params_slice, min_name: str, max_name: str, train=True, pred=True, device='cuda:0') -> None:
        super().__init__(train, pred, device)
        self.min_ = params[min_name][:, None, params_slice]
        self.max_ = params[max_name][:, None, params_slice]
        self.slice = torch.LongTensor(slice).to(device)
    
    def __call__(self, data: Tensor):
        data[:, :, self.slice] = torch.clip(data.index_select(2, self.slice), self.min_, self.max_)


class StandardScaler(ProcessorBase):
    """
    Input:
        data (timesteps, tickers, feature) = (L, N, F)
        mean (timesteps, feature) = (L, F)
        std (timesteps, feature) = (L, F)
    Function: scale the certain features set (slice) by mean and std parameters (must be 2-D tensor) provided
    """
    def __init__(self, params: Dict[str, Tensor], slice, params_slice, mean_name: str, std_name: str, train=True, pred=True, device='cuda:0', eps=1e-8) -> None:
        super().__init__(train, pred, device)
        self.mean = params[mean_name][:, None, params_slice]
        self.std = params[std_name][:, None, params_slice]
        self.slice = torch.LongTensor(slice).to(device)
        self.eps = eps
    
    def __call__(self, data: Tensor):
        data[:, :, self.slice] = (data.index_select(2, self.slice) - self.mean) / (self.std + self.eps)


# %% dataset processor
class DatasetProcessor:

    def __init__(self, dataset: DefaultDataset = None, max_ele_per_group: int = 300) -> None:
        self.dataset = dataset
        self.max_ele_per_group = max_ele_per_group
    
    def init(self, start_date: int = None, end_date: int = None, preprocesses: List[Dict[str, Any]] = [], postprocesses: List[Dict[str, Any]] = []):
        self.init_preprocess(preprocesses)
        self.fit(start_date, end_date)
        self.init_postprocess(postprocesses)
    
    def init_preprocess(self, preprocesses: List[Dict[str, Any]] = []):
        self.preprocesses = []
        for process in preprocesses:
            for key, params in process['params'].items():
                if 'slice' in key:
                    process['params'][key] = self.dataset.get_slice(**params)
            process: ProcessorBase = process['class'](**process['params'], device=self.dataset.device)
            self.preprocesses.append(process)
    
    def preprocess(self, data: Tensor) -> Tensor | None:
        for process in self.preprocesses:
            if data is None:
                return
            if 0 in data.size():
                return
            if (self.train and process.train) or (self.pred and process.pred):
                data = process(data)
        return data
    
    def fit(self, start_date: int = None, end_date: int = None):
        stt_ix = max(self.dataset.window, np.where(self.dataset.data_engine.trade_dates >= start_date)[0][0])
        end_ix = np.where(self.dataset.data_engine.trade_dates <= end_date)[0][-1]
        slice_ = self.dataset.get_slice(type = ['feature'])
        max_per_group = max(1, int(self.max_ele_per_group / self.dataset.window))
        ngroups = len(slice_) // max_per_group + int(len(slice_) % max_per_group != 0)
        groups = np.pad(slice_, (0, max_per_group * ngroups - len(slice_)), constant_values=-1).reshape(ngroups, max_per_group).tolist()
        for i in range(len(groups)):
            while -1 in groups[i]:
                groups[i].remove(-1)
        params = []
        for i in range(len(groups)):
            slice_group = groups[i]
            print(f"Fitting params for [{i + 1}th] group of features: {slice_group} ...")
            dataset_for_fit = []
            for ix in tqdm(range(stt_ix, end_ix + 1), ncols=80, desc=f'{start_date}-{end_date}'):
                batch = self.dataset.batching(ix)
                batch = self.preprocess(batch)
                if batch is None:
                    continue
                if batch.size(1) < 100:
                    continue
                batch = batch[:, :, slice_group].transpose(0, 1)
                dataset_for_fit.append(batch.cpu())
                torch.cuda.empty_cache()
            dataset_for_fit = torch.cat(dataset_for_fit)
            print(f'Size of the whole set of [{i + 1}th] group of features = {dataset_for_fit.size()}.')
            param = self._fit_params(dataset_for_fit)
            params.append(param)
            for _ in range(10):
                torch.cuda.empty_cache()
                time.sleep(0.1)
        params = {k: torch.cat([p[k] for p in params], dim=-1).to(self.dataset.device) for k in params[0]}
        self.params = param
    
    def _fit_params(self, data: Tensor) -> Dict[str, Tensor]:
        gc.collect()
        median = torch.nanmedian(data, dim=0).values
        gc.collect()
        mad = torch.nanmedian(torch.abs(data - median), dim=0).values
        gc.collect()
        mean = torch.nanmean(data, dim=0)
        gc.collect()
        std = torch.std(data, dim=0)
        gc.collect()
        mstd = 1.483 * mad
        min_ = median - 5 * mstd
        max_ = median + 5 * mstd
        return {
            'median': median,
            'mad': mad,
            'mean': mean,
            'std': std,
            'mstd': mstd,
            'min': min_,
            'max': max_
        }
    
    def init_postprocess(self, postprocesses: List[Dict[str, Any]]):
        self.postprocesses = []
        for process in postprocesses:
            this_params = copy.deepcopy(process['params'])
            for key, params in process['params'].items():
                if 'slice' in key:
                    if isinstance(params, Mapping):
                        feature_slice = self.dataset.get_slice(type = ['feature'])
                        this_slice = self.dataset.get_slice(**params)
                        params_slice = [feature_slice.index(i) for i in this_slice]
                        this_params[key] = this_slice
                        this_params[f'params_{key}'] = params_slice  # 确定data_engine中的slice对应到params中的位置
            process: ProcessorBase = process['class'](params=self.params, device=self.dataset.device, **this_params)
            self.postprocesses.append(process)
    
    def postprocess(self, data: Tensor | None) -> Tensor | None:
        for process in self.postprocesses:
            if data is None:
                continue
            if 0 in data.size():
                continue
            if (self.train and process.train) or (self.pred and process.pred):
                data = process(data)
        return data

    def process(self, data: Tensor) -> Tensor | None:
        data = self.preprocess(data)
        data = self.postprocess(data)
        if data is None:
            return
        if 0 in data.size():
            return
        data = data.transpose(0, 1)
        out = {key: data.index_select(2, self.dataset.splits_slice[key]) for key in self.dataset.splits}
        out['trade_date'] = data[:, -1, -2]
        out['ticker'] = data[:, -1, -1]
        return out
    
    def training(self):
        self.train = True
        self.pred = False
    
    def eval(self):
        self.train = False
        self.pred = True
    
    def __call__(self, data: Tensor) -> Tensor | None:
        return self.process(data)