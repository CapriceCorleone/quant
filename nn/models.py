'''
Author: WangXiang
Date: 2024-04-19 22:50:26
LastEditTime: 2024-04-20 15:44:00
'''

import os
import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from ..core import Aligner


class FinancialConfig(PretrainedConfig):

    model_type = 'finalcial'


class PreTrainedModelManager:

    def __init__(self) -> None:
        pass

    def init_model(
        self,
        model_class: PreTrainedModel,
        configuration_dict: Dict = None,
        model_directory: str = None,
        **kwargs
    ):
        if configuration_dict is not None:
            print('Model will be randomly initialized by model configuration provided.')
            config = model_class.config_class.from_dict(configuration_dict)
            model = model_class(config, **kwargs)
            if model_directory is not None:
                print(f"Model configuration and parameters will be saved to {model_directory}.")
                model.save_pretrained(model_directory, safe_serialization=False)
        else:
            print(f"Model will be initialized from the configuration and parameters in {model_directory}.")
            model = model_class.from_pretrained(model_directory, **kwargs)
        return model


def match_and_load_state_dict(cls: PreTrainedModel):
    state_dict_path = Path(cls.config._name_or_path, 'pytorch_model.bin')
    if state_dict_path.exists():
        print('Loading saved parameters ... ', end = '')
        state_dict = torch.load(state_dict_path, map_location=cls.device)
        try:
            cls.load_state_dict(state_dict, strict=False)
            print('success loading.')
        except RuntimeError:
            print('unable to load saved parameters.')


# %% NN model
class RNNCell(PreTrainedModel):

    config_class = FinancialConfig

    def __init__(self, config: PretrainedConfig, device, *inputs, **kwargs):
        super().__init__(config)
        self.config = config
        self.device = device

        rnn_type = config.rnn_type if config.rnn_type is not None else 'gru'
        rnn_input_size = config.rnn_input_size
        rnn_hidden_size = config.rnn_hidden_size
        num_rnn_layers = config.num_rnn_layers
        bidirectional = config.bidirectional
        scale = 2 if bidirectional else 1
        dropout_prob = config.dropout_prob

        if rnn_type == 'gru':
            rnn_func = nn.GRU
        elif rnn_type == 'lstm':
            rnn_func = nn.LSTM
        else:
            raise TypeError(f"No RNN module named as {rnn_type}, please provide either 'gru' or 'lstm'.")
        self.rnn = rnn_func(
            input_size    = rnn_input_size,
            hidden_size   = rnn_hidden_size,
            num_layers    = num_rnn_layers,
            batch_first   = True,
            bidirectional = bidirectional
        )
        self.batch_norm = nn.BatchNorm1d(rnn_hidden_size * scale)
        self.dropout = nn.Dropout(p=dropout_prob)

        match_and_load_state_dict(self)

    def batch_forward(self, inputs: Tensor = None, labels: Tensor = None, risks: Tensor = None, ticker: Tensor = None, trade_date: Tensor = None, **kwargs):
        labels = labels[:, -1].detach()
        risks = risks[:, -1].detach()
        hidden, _ = self.rnn(inputs)
        hidden = self.dropout(self.batch_norm(hidden[:, -1, :]))
        logits = hidden.mean(dim=-1)
        risks = risks[:, risks.abs().sum(dim=0) != 0]
        logits = logits - risks @ torch.linalg.inv(risks.T @ risks) @ risks.T @ logits
        return {'labels': labels, 'logits': logits, 'hidden': hidden, 'ticker': ticker, 'trade_date': trade_date}
    
    def forward(self, data: Dict[str, Tensor] = None, **kwargs):
        NULL_OUTPUT = {'labels': None, 'logits': None, 'hidden': None, 'ticker': None, 'trade_date': None}
        if data is None:
            return NULL_OUTPUT
        return self.batch_forward(**data, **kwargs)


class SelfAttention(nn.Module):

    def __init__(self, input_size: int, attn_hidden_size: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.attn_hidden_size = attn_hidden_size
        self.num_heads = num_heads
        self.attn_head_size = int(self.attn_hidden_size / self.num_heads)
        self.all_head_size = self.num_heads * self.attn_head_size
        self.query = nn.Linear(self.input_size, self.all_head_size)
        self.key = nn.Linear(self.input_size, self.all_head_size)
        self.value = nn.Linear(self.input_size, self.all_head_size)
        self.downsample = nn.Linear(self.input_size, self.all_head_size) if self.input_size != self.all_head_size else None
        self.linear = nn.Linear(self.all_head_size, self.all_head_size)
        self.layer_norm = nn.LayerNorm((self.all_head_size, ))
        self.dropout = nn.Dropout(dropout)

    def multi_head_transpose(self, x: Tensor) -> Tensor:
        if self.num_heads == 1:
            return x
        shape = x.size()[:-1] + (self.num_heads, self.attn_head_size)
        x = x.view(shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x: Tensor) -> Tensor:
        shape = x.size()
        query_layer = self.multi_head_transpose(self.query(x))
        key_layer = self.multi_head_transpose(self.key(x))
        value_layer = self.multi_head_transpose(self.value(x))
        score = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        score = F.softmax(score, dim=-1) / self.attn_head_size ** 0.5
        att = torch.matmul(score, value_layer)
        if self.num_heads > 1:
            att = att.permute(0, 2, 1, 3).contiguous().view(shape[:-1] + (self.all_head_size, ))
        res = x if self.downsample is None else self.downsample(x)
        out = self.dropout(self.layer_norm(self.linear(att + res)))
        return out


class ARNNCell(PreTrainedModel):

    config_class = FinancialConfig

    def __init__(self, config: PretrainedConfig, device, *inputs, **kwargs):
        super().__init__(config)
        self.config = config
        self.device = device

        rnn_type = config.rnn_type if config.rnn_type is not None else 'gru'
        rnn_input_size = config.rnn_input_size
        rnn_hidden_size = config.rnn_hidden_size
        num_rnn_layers = config.num_rnn_layers
        bidirectional = config.bidirectional
        scale = 2 if bidirectional else 1
        dropout_prob = config.dropout_prob

        if rnn_type == 'gru':
            rnn_func = nn.GRU
        elif rnn_type == 'lstm':
            rnn_func = nn.LSTM
        else:
            raise TypeError(f"No RNN module named as {rnn_type}, please provide either 'gru' or 'lstm'.")
        self.rnn = rnn_func(
            input_size    = rnn_input_size,
            hidden_size   = rnn_hidden_size,
            num_layers    = num_rnn_layers,
            batch_first   = True,
            bidirectional = bidirectional
        )
        self.attn_net = ...
        self.batch_norm = nn.BatchNorm1d(rnn_hidden_size * scale)
        self.dropout = nn.Dropout(p=dropout_prob)

        match_and_load_state_dict(self)

    def batch_forward(self, inputs: Tensor = None, labels: Tensor = None, risks: Tensor = None, ticker: Tensor = None, trade_date: Tensor = None, **kwargs):
        labels = labels[:, -1].detach()
        risks = risks[:, -1].detach()
        hidden, _ = self.rnn(inputs)
        hidden = self.attn_net(hidden)
        hidden = self.dropout(self.batch_norm(hidden[:, -1, :]))
        logits = hidden.mean(dim=-1)
        risks = risks[:, risks.abs().sum(dim=0) != 0]
        logits = logits - risks @ torch.linalg.inv(risks.T @ risks) @ risks.T @ logits
        return {'labels': labels, 'logits': logits, 'hidden': hidden, 'ticker': ticker, 'trade_date': trade_date}
    
    def forward(self, data: Dict[str, Tensor] = None, **kwargs):
        NULL_OUTPUT = {'labels': None, 'logits': None, 'hidden': None, 'ticker': None, 'trade_date': None}
        if data is None:
            return NULL_OUTPUT
        return self.batch_forward(**data, **kwargs)


# %% ML model
class LightGBM:

    def __init__(self) -> None:
        self.aligner = Aligner()

    def load_data(self, filenames: List[str | Path]) -> Tuple[np.ndarray, np.ndarray]:
        print('Loading data from following files:')
        data = []
        for i, fn in enumerate(filenames):
            print('\t', fn)
            dfs: Dict[str, pd.DataFrame] = pd.read_pickle(fn)
            if i == 0:
                min_date = dfs['labels'].index.min()
                max_date = dfs['labels'].index.max() + 10000
                data.append(self.aligner.align(dfs['labels']).loc[min_date:max_date].values)
                tdays = self.aligner.align(dfs['labels']).loc[min_date:max_date].index.values
            dfs.pop('labels')
            gc.collect()
            keys = list(dfs.keys())
            for key in keys:
                data.append(self.aligner.align(dfs[key]).loc[min_date:max_date].values)
                dfs.pop(key)
                gc.collect()
        gc.collect()
        data = np.stack(data)
        gc.collect()
        data = data.transpose(1, 2, 0)
        return data, tdays
    
    def preprocess(self, X: np.ndarray, y: np.ndarray, univ: np.ndarray, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X = np.where(univ[:, :, None] == 1, X, np.nan)
        y = np.where(univ == 1, y, np.nan)
        md = np.nanmedian(y)
        mad = np.nanmedian(np.abs(y - md))
        min_ = md - 3 * mad
        max_ = md + 3 * mad
        y = np.clip(y, min_, max_)
        y = (y - np.nanmean(y, axis=1)[:, None]) / (np.nanstd(y, axis=1)[:, None] + 1e-8)
        if train:
            mask = (np.abs(y) > 3) | np.isnan(y) | np.isnan(X).all(axis=2)
            X = np.where(mask[:, :, None], np.nan, X)
            y = np.where(mask, np.nan, y)
        return X, y
    
    def get_desc(self, days: List[int], X: np.ndarray, y: np.ndarray) -> str:
        ndays = len(days)
        ntickers = int((~np.isnan(y)).sum(axis=1).mean())
        stt = days[0]
        end = days[1]
        return f'{stt}-{end}: {ndays:d}(d)*{ntickers:d}(s)'
    
    def reshape(self, X: np.ndarray, y: np.ndarray, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        T, N, F = X.shape
        X = X.reshape(T * N, F)
        y = y.reshape(T * N)
        if train:
            mask = np.isnan(y)
            return X[~mask], y[~mask]
        return X, y
    
    def train(
        self,
        periods: Dict[int, Dict[str, List[np.ndarray] | np.ndarray]],
        data: np.ndarray,
        tdays: np.ndarray,
        train_universe: pd.DataFrame,
        pred_universe: pd.DataFrame,
        init_date: int,
        saved_outputs_dir: List[str | Path],
        use_saved_models: bool,
        params: Dict[str, Any]
    ):
        saved_models_dir = Path(saved_outputs_dir, 'models')
        os.makedirs(saved_models_dir, exist_ok=True)
        if use_saved_models:
            saved_models_filenames = [Path(saved_models_dir, i) for i in os.listdir(saved_models_dir)]
            saved_factor_filename = Path(saved_outputs_dir, 'factor.pkl')
            if saved_factor_filename.exists():
                saved_factor = pd.read_pickle(saved_factor_filename)
                saved_factor_max_date = saved_factor.index.max()
            else:
                saved_factor = None
                saved_factor_max_date = 20050101
        else:
            saved_factor = None
            saved_factor_filename = Path(saved_outputs_dir, 'factor.pkl')
            saved_models_filenames = []
        
        print('-' * 110)
        print(f"{'Train Data':^34} | {'Test Data':^34} | {'Train Loss':^10} | {'Test Loss':^10} | {'Elapsed':^7}")
        print('-' * 110)
        start0 = time.time()
        outputs = []

        for day, period in periods.items():
            start1 = time.time()
            model_file = Path(saved_models_dir, f"{period['train'][1][-1]}.txt")
            if period['test'][0] < tdays[0]:
                continue
            if period['test'][-1] < init_date:
                continue
            
            if use_saved_models:
                if period['test'][-1] < saved_factor_max_date:
                    continue
                if model_file in saved_models_filenames:
                    do_train = False
                else:
                    do_train = True
            else:
                do_train = True
            
            if do_train:
                train_days = np.intersect1d(tdays, period['train'][0])
                if len(train_days) == 0:
                    train_data_desc = ''
                    train_loss_desc = ''
                    continue
                train_univ = train_universe.loc[train_days].values
                train_days_ix = np.sort([np.where(tdays == i)[0][0] for i in train_days])
                if len(train_days_ix) == train_days_ix.max() - train_days_ix.min() + 1:
                    X_train = data[train_days_ix.min() : train_days_ix.max() + 1, :, 1:]
                    y_train = data[train_days_ix.min() : train_days_ix.max() + 1, :, 0]
                else:
                    X_train = data[train_days_ix, :, 1:]
                    y_train = data[train_days_ix, :, 0]
                X_train, y_train = self.preprocess(X_train, y_train, train_univ, train=True)
                train_data_desc = self.get_desc(train_days, X_train, y_train)
                X_train, y_train = self.reshape(X_train, y_train, train=True)
                if len(X_train) == 0:
                    train_loss_desc = ''
                    continue
                train_set = lgb.Dataset(X_train, y_train, free_raw_data=False)
                params['feature_fraction'] = min(0.9 / 200 / X_train.shape[1])
                params['monotone_constraints'] = [1] * X_train.shape[1]
                num_boost_round = params.pop('num_boost_round', 200)
                model = lgb.train(params, train_set, num_boost_round=num_boost_round)
                model.save_model(model_file)
                y_pred = model.predict(X_train)
                try:
                    mse = mean_squared_error(y_train, y_pred)
                except ValueError:
                    mse = -1.
                train_loss_desc = f'{mse:.6f}'
                del X_train, y_train; gc.collect()
            else:
                model = lgb.Booster(model_file=model_file)
                train_data_desc = ''
                train_loss_desc = ''
            
            test_days = np.intersect1d(tdays, period['test'])
            test_days = test_days[test_days >= init_date]
            if len(test_days) == 0:
                continue
            pred_univ = pred_universe.loc[test_days].values
            test_days_ix = np.array([np.where(tdays == i)[0][0] for i in test_days])
            X_test = data[test_days_ix, :, 1:]
            y_test = data[test_days_ix, :, 0]
            X_test, y_test = self.preprocess(X_test, y_test, pred_univ, train=False)
            test_data_desc = self.get_desc(test_days, X_test, y_test, train=False)
            X_test, y_test = self.reshape(X_test, y_test, train=False)
            y_pred = model.predict(X_test)
            mask = np.isnan(X_test).all(axis=1)
            y_pred[mask] = np.nan
            try:
                mse = mean_squared_error(y_test[~np.isnan(y_test)], y_pred=[~np.isnan(y_test)])
            except ValueError:
                mse = -1.
            test_loss_desc = f'{mse:.6f}'
            del X_test, y_test; gc.collect()

            y_pred = y_pred.reshape(len(test_days), -1)
            y_pred = pd.DataFrame(y_pred, index=test_days, columns=self.aligner.tickers)
            outputs.append(y_pred)
            elapsed_desc = f'{time.time() - start1:.2f}'
            print(f'{train_data_desc:^34} | {test_data_desc:^34} | {train_loss_desc:^10} | {test_loss_desc:^10} | {elapsed_desc:^7}')
        
        outputs = pd.concat(outputs, axis=0)
        if saved_factor is not None:
            outputs = pd.concat([
                saved_factor[saved_factor.index < outputs.index.min()],
                outputs
            ], axis=0)
        outputs.to_pickle(saved_factor_filename)
        print('-' * 110)
        print(f'Total elapsed time: {time.time() - start0:.2f}')
        print(f'{datetime.now()}: Saving output to {saved_factor_filename}.')
        print('\tLast date: ', outputs.index.max())
        return outputs
    
    def compose(self, saved_outputs_dirs: str | Path) -> pd.DataFrame:
        composed_factor = None
        for saved_outputs_dir in saved_outputs_dirs:
            factor = pd.read_pickle(Path(saved_outputs_dir, 'factor.pkl'))
            factor = factor.dropna(how='all')
            factor = factor[factor.index >= int(Path(saved_outputs_dir).name)]
            if len(factor) == 0:
                continue
            if composed_factor is None:
                composed_factor = factor
            else:
                composed_factor = pd.concat([
                    composed_factor[composed_factor.index < factor.index.min()],
                    factor
                ], axis=0)
        return composed_factor
    
    def compose_equal(self, saved_outputs_dir: str | Path) -> pd.DataFrame:
        saved_outputs_dir = Path(saved_outputs_dir)
        composed_factor = None
        for saved_output_file in os.listdir(saved_outputs_dir):
            factor = pd.read_pickle(saved_outputs_dir / saved_output_file)
            factor = factor.dropna(how='all')
            filename = saved_output_file.split('.')[0]
            if not filename.isdigit():
                continue
            factor = factor[factor.index >= int(saved_output_file.split('.')[0])]
            if len(factor) == 0:
                continue
            if composed_factor is None:
                composed_factor = factor
            else:
                composed_factor = pd.concat([
                    composed_factor[composed_factor.index < factor.index.min()],
                    factor
                ], axis=0)
        return composed_factor
    
    def save_factor(self, factor: pd.DataFrame, filename: str | Path):
        filename = Path(filename)
        if filename.exists():
            old = pd.read_pickle(filename)
            factor = pd.concat([
                old[old.index < factor.index.min()],
                factor
            ], axis=0)
        factor.to_pickle(filename)
        print(f'{datetime.now()}: Saving factor to {filename}.')
        print('\tLast date: ', factor.index.min())
    
    def run(
        self,
        periods: Dict[int, Dict[str, List[np.ndarray] | np.ndarray]],
        train_universe: pd.DataFrame,
        pred_universe: pd.DataFrame,
        saved_inputs_dirs: List[str | Path],
        saved_outputs_dir: str | Path,
        init_date: int,
        use_saved_models: bool,
        params: Dict[str, Any]
    ):
        saved_inputs_filenames = [[Path(saved_inputs_dir, i) for i in os.listdir(saved_inputs_dir)] for saved_inputs_dir in saved_inputs_dirs]
        assert all([len(i) == len(j) for i in saved_inputs_filenames for j in saved_inputs_filenames])
        saved_outputs_dirs = []
        for i in range(len(saved_inputs_filenames[0])):
            cur_saved_inputs_filenames = [j[i] for j in saved_inputs_filenames]
            val_end_date = int(cur_saved_inputs_filenames[0].name[:-4])
            if val_end_date < init_date:
                continue
            data, tdays = self.load_data(cur_saved_inputs_filenames)
            cur_saved_outputs_dir = Path(saved_outputs_dir, str(val_end_date))
            self.train(
                periods           = periods,
                data              = data,
                tdays             = tdays,
                train_universe    = train_universe,
                pred_universe     = pred_universe,
                init_date         = init_date,
                saved_outputs_dir = saved_outputs_dir,
                use_saved_models  = use_saved_models,
                params            = params
            )
            saved_inputs_dirs.append(cur_saved_outputs_dir)
            del data; gc.collect()
        factor = self.compose(saved_outputs_dirs)
        filename = Path(saved_outputs_dir, 'factor.pkl')
        self.save_factor(factor, filename)
        if len(saved_inputs_dirs) == 1:
            equal_factor = self.compose_equal(Path(saved_outputs_dir).parent / 'logits')
            filename = Path(saved_outputs_dir).parent / 'logits/factor.pkl'
            self.save_factor(equal_factor, filename)