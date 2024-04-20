'''
Author: WangXiang
Date: 2024-04-20 15:33:00
LastEditTime: 2024-04-20 15:33:12
'''

import os
import yaml
import pickle
import warnings
import pandas as pd
from typing import List
from pathlib import Path
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

import torch

from .trainer import get_last_checkpoint


def join(loader, node):
    seq = loader.construct_sequence(node)
    return '-'.join([str(i) for i in seq])


def join_path(loader, node):
    seq = loader.construct_sequence(node)
    return Path(*[str(i) for i in seq])


def ranger(loader, node):
    seq = loader.construct_sequence(node)
    return list(range(seq[0]))


def expr(loader, node):
    seq = loader.construct_sequence(node)
    seq = list(map(str, seq))
    return eval(''.join(seq))


class TaskNN:

    def __init__(self, configure_files: List[str]):
        yaml.add_constructor('!join', join)
        yaml.add_constructor('!join_path', join_path)
        yaml.add_constructor('!ranger', ranger)
        yaml.add_constructor('!expr', expr)
        self.configures = self.load_configures(configure_files)
        self.last_nn_period = None

    def load_configures(self, configure_files):
        configures = []
        for config_file in configure_files:
            with open(config_file, 'r', encoding='utf-8') as file:
                config = yaml.load(file, yaml.FullLoader)
            configures.append(config)
        return configures

    def setup_torch_environments(self, config, last_config=defaultdict(dict)):
        if config['torch_backends'] == last_config['torch_backends']:
            print('The same torch environments has been set, skip this node.')
            return
        torch_num_threads = config['torch_backends']['torch_num_threads']
        torch.set_num_threads(torch_num_threads)
        print(f'{datetime.now()}: Set torch number of threads to {torch_num_threads}.')
        torch_num_interop_threads = config['torch_backends']['torch_num_interop_threads']
        torch.set_num_interop_threads(torch_num_interop_threads)
        print(f'{datetime.now()}: Set torch number of interop threads to {torch_num_interop_threads}.')
    
    def create_data_engine(self, config, last_config=defaultdict(dict)):
        # 创建DataEngine实例
        if config['data_engine'] == last_config['data_engine']:
            print('The same data engine has been created, skip this node.')
            return
        self.data_engine = config['data_engine']['class'](**config['data_engine']['params'])
        self.data_engine.prepare()
        print(f'{datetime.now()}: Create data engine.')
        print('\tData Engine: ' + '\n\t'.join(str(self.data_engine).split('\n')))
    
    def generate_periods(self, config, last_config=defaultdict(dict)):
        # 获取训练周期（训练集/验证集/测试集）划分
        if config['calendar'] == last_config['calendar']:
            print('The same periods has been generated, skip this node.')
            return
        self.calendar = config['calendar']['class']()
        self.nn_periods = self.calendar.cut_periods(**config['calendar']['NN'])
        self.nn_simplified_periods = self.calendar.simplify_periods(self.nn_periods)

    def generate_universe(self, config, last_config=defaultdict(dict)):
        # 创建训练集/测试集股票池
        if config['universe'] == last_config['universe']:
            print('The same universes has been generated, skip this node.')
            return
        self.univ = config['universe']['class']()
        self.train_universe = self.univ(**config['universe']['train'])
        self.pred_universe = self.univ(**config['universe']['pred'])
        print(f'{datetime.now()}: Create train and predict universe.')
        print('\tTrain Universe: '  + '\n\t'.join(str(self.train_universe).split('\n')))
        print('\tPredict Universe: ' + '\n\t'.join(str(self.pred_universe).split('\n')))

    def create_train_dataset(self, nn_period, config, last_config=defaultdict(dict)):
        # 训练集
        # 创建Dataset实例（这个Dataset不是torch.utils.data.Dataset，而是本框架中自定义的Dataset）
        if nn_period == self.last_nn_period and config['dataset'] == last_config['dataset']:
            return
        self.train_dataset = config['dataset']['class'](
            self.data_engine,
            start_date = nn_period['train'][0][0],
            end_date   = nn_period['train'][0][-1],
            universe   = self.train_universe,
            **config['dataset']['params']
        )
        print(f'{datetime.now()}: Create train dataset.')
        print('\tTrain Dataset: ', self.train_dataset)

    def create_and_fit_dataset_processor(self, nn_period, config, last_config=defaultdict(dict)):
        # 创建DatasetProcessor实例，进行面板标准化参数估计
        if nn_period == self.last_nn_period and config['dataset'] == last_config['dataset']:
            return
        self.dataset_processor = config['dataset_processor']['class'](
            self.train_dataset,
            **config['dataset_processor']['params']
        )
        print(f'{datetime.now()}: Create dataset processor and start parameters fitting ...')
        self.dataset_processor.training()
        self.dataset_processor.init(
            start_date = nn_period['train'][0][0],
            end_date   = nn_period['train'][0][-1],
            **config['dataset_processor']['init_params']
        )
        print(f'{datetime.now()}: Parameters fitting completed.')
    
    def create_eval_dataset(self, nn_period, config, last_config=defaultdict(dict)):
        # 验证集
        # 创建Dataset实例
        if nn_period == self.last_nn_period and config['dataset'] == last_config['dataset']:
            return
        self.eval_dataset = config['dataset']['class'](
            self.data_engine,
            start_date = nn_period['val'][0][0],
            end_date   = nn_period['val'][0][-1],
            universe   = self.train_universe,
            **config['dataset']['params']
        )
        print(f'{datetime.now()}: Create eval dataset.')
        print('\tEval Dataset: ', self.eval_dataset)

    def create_test_dataset(self, nn_period, config, last_config):
        # 预测集
        # 创建Dataset实例
        if nn_period == self.last_nn_period and config['dataset'] == last_config['dataset']:
            return
        if config['dataset'].get('init_date', None) is None:
            start_date = nn_period['test'][0]
        else:
            start_date = max(nn_period['test'][0], config['dataset']['init_date'])
        self.test_dataset = config['dataset']['class'](
            self.data_engine,
            start_date = start_date,
            end_date   = nn_period['test'][-1],
            universe   = self.pred_universe,
            **config['dataset']['params']
        )
        print(f'{datetime.now()}: Create test dataset.')
        print('\tTest Dataset: ', self.test_dataset)

    def relocate_data_engine(self, config, last_config=defaultdict(dict)):
        # 判断是否需要变更DataEngine的最终存储位置
        if config['data_engine'] == last_config['data_engine']:
            return
        data_engine_final_device = config['data_engine'].get('final_device', None)
        if data_engine_final_device is not None:
            self.data_engine.to(data_engine_final_device)
        print(f'{datetime.now()}: Data engine is located at [{self.data_engine.device}].')

    def create_pretrained_model_manager(self, config, last_config=defaultdict(dict)):
        # 创建PretrainedModelManager实例
        if config['pretrained_model_manager'] == last_config['pretrained_model_manager']:
            return
        self.pretrained_model_manager = config['pretrained_model_manager']['class']()

    def initialize_model(self, config):
        # 模型初始化
        self.model = self.pretrained_model_manager.init_model(
            **config['model']
        )
        print(f'{datetime.now()}: Initialize model.')
        print('\t' + '\n\t'.join(str(self.model).split('\n')))

    def initialize_train_args(self, nn_period, config):
        # 训练参数初始化
        kwargs = config['train_args']['params'].copy()
        output_dir = Path(
            kwargs.pop('output_dir'),
            str(nn_period['val'][1][-1])
        )
        self.train_args = config['train_args']['class'](
            output_dir = output_dir,
            **kwargs
        )
        print(f'{datetime.now()}: Set training arguments.')
        print('\tTraining Arguments: ' + '\n\t'.join(str(self.train_args).split('\n')))
    
    def initialize_optimizer(self, config):
        # 优化器初始化
        self.optimizer = config['optimizer']['class'](
            params       = self.model.parameters(),
            lr           = self.train_args.learning_rate,
            weight_decay = self.train_args.weight_decay
        )
        print(f'{datetime.now()}: Initialize optimizer.')
        print('\tOptimizer: ' + '\n\t'.join(str(self.optimizer).split('\n')))

    def initialize_lr_scheduler(self, config):
        # 学习率机制初始化
        self.scheduler = config['scheduler']['class'](
            optimizer = self.optimizer,
            **config['scheduler']['params']
        )
        print(f'{datetime.now()}: Initialize learning rate scheduler.')
        print('\tScheduler: ', self.scheduler)

    def initialize_loss_function(self, config):
        # 损失函数初始化
        self.loss_fn = config['loss_fn']['class'](
            **config['loss_fn']['params']
        )
        print(f'{datetime.now()}: Initialize loss function.')
        print('\tLoss function: ', self.loss_fn)

    def initialize_trainer(self, config):
        # 训练器初始化
        self.trainer = config['trainer']['class'](
            model             = self.model,
            args              = self.train_args,
            train_dataset     = self.train_dataset,
            dataset_processor = self.dataset_processor,
            eval_dataset      = self.eval_dataset,
            optimizers        = (self.optimizer, self.scheduler),
            loss_fn           = self.loss_fn,
            **config['trainer'].get('params', {})
        )
        print(f'{datetime.now()}: Initialize trainer.')
        print('\tTrainer: ', self.trainer)

    def load_model(self, nn_period, config):
        # 加载模型（推理时使用）
        train_output_dir = Path(
            config['inference']['train_output_dir'],
            str(nn_period['val'][1][-1])
        )
        checkpoint = get_last_checkpoint(train_output_dir)
        print(f'{datetime.now()}: Load model from {checkpoint} for inference.')
        self.model = config['model']['model_class'].from_pretrained(
            checkpoint,
            device = config['model']['device'],
        )

    def initialize_train_args_for_inference(self, config):
        # 训练参数初始化（推理时使用）
        self.train_args = config['inference']['train_args']['class'](
            **config['inference']['train_args']['params']
        )

    def initialize_trainer_for_inference(self, config):
        # 训练器初始化（推理时使用）
        self.trainer = config['inference']['trainer']['class'](
            model             = self.model,
            args              = self.train_args,
            train_dataset     = None,
            dataset_processor = self.dataset_processor,
            eval_dataset      = None,
            optimizers        = (None, None),
            loss_fn           = None,
        )

    def save_logits(self, nn_period, config, logits):
        save_path = Path(config['inference']['output_dir']['logits'], f"{nn_period['val'][1][-1]}.pkl")
        logits['trade_date'] = self.data_engine.trade_dates[logits['trade_date'].values]
        logits['ticker'] = self.data_engine.tickers[logits['ticker'].values]
        logits = logits.pivot(index='trade_date', columns='ticker', values='logits')
        self._save(save_path, logits)

    def save_hidden(self, nn_period, config, hidden):
        save_path = Path(config['inference']['output_dir']['hidden'], f"{nn_period['val'][1][-1]}.pkl")
        hidden['trade_date'] = self.data_engine.trade_dates[hidden['trade_date'].values]
        hidden['ticker'] = self.data_engine.tickers[hidden['ticker'].values]
        hidden = hidden.set_index(['trade_date', 'ticker']).unstack()
        self._save(save_path, hidden)

    def _save(self, path, data):
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        if isinstance(data.columns, pd.MultiIndex):
            data = {key: data[key] for key in data.columns.levels[0]}
            self._save_pickle(path, data)
        else:
            self._save_dataframe(path, data)

    def _save_dataframe(self, path, data):
        path = Path(path)
        if path.exists():
            old = pd.read_pickle(path)
            old = old[old.index < data.index.min()]
            data = pd.concat([old, data], axis=0)
        data.to_pickle(path)

    def _save_pickle(self, path, data):
        path = Path(path)
        if path.exists():
            old = pd.read_pickle(path)
            keys = list(set(list(old.keys()) + list(data.keys())))
            for key in keys:
                if key in old.keys() and key in data.keys():
                    df = old[key]
                    df = df[df.index < data[key].index.min()]
                    data[key] = pd.concat([df, data[key]], axis=0)
                elif key in old.keys():
                    data[key] = old[key]
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def train_nn(self, year, nn_period, config, last_config=defaultdict(dict)):
        nn_simplified_period = self.nn_simplified_periods[year]
        print('Current training period:')
        for k, v in nn_simplified_period.items():
            print(f'\t{k:^5}:', v)
        self.create_train_dataset(nn_period, config, last_config)
        self.create_and_fit_dataset_processor(nn_period, config, last_config)
        self.create_eval_dataset(nn_period, config, last_config)
        self.relocate_data_engine(config, last_config)
        self.create_pretrained_model_manager(config, last_config)
        self.initialize_model(config)
        self.initialize_train_args(nn_period, config)
        self.initialize_optimizer(config)
        self.initialize_lr_scheduler(config)
        self.initialize_loss_function(config)
        self.initialize_trainer(config)
        # 开始训练
        print('\n')
        if not config['skip_train']:
            self.trainer.train(config=config)

    def infer_nn(self, year, nn_period, config, last_config=defaultdict(dict)):
        nn_simplified_period = self.nn_simplified_periods[year]
        print('Current training and inference period:')
        for k, v in nn_simplified_period.items():
            print(f'\t{k:^5}:', v)
        self.create_test_dataset(nn_period, config, last_config)
        if config['inference']['fit_dataset_processor']:
            self.create_and_fit_dataset_processor(nn_period, config, last_config)
            self.relocate_data_engine(config, last_config)
        self.load_model(nn_period, config)
        self.initialize_train_args_for_inference(config)
        self.initialize_trainer_for_inference(config)
        # 开始推理
        self.dataset_processor.eval()
        logits, hidden = self.trainer.predict(self.test_dataset)
        self.save_logits(nn_period, config, logits)
        self.save_hidden(nn_period, config, hidden)
        print('\n\n\n\n')

    def change_model_id(self, config, repeat):
        old_model_id = config['model']['model_id']
        new_model_id = f'model_{repeat}'
        config['model']['model_id'] = new_model_id
        config['train_args']['params']['output_dir'] = Path(config['train_args']['params']['output_dir'].as_posix().replace(old_model_id, new_model_id))
        config['inference']['train_output_dir'] = Path(config['inference']['train_output_dir'].as_posix().replace(old_model_id, new_model_id))
        config['inference']['output_dir']['logits'] = Path(config['inference']['output_dir']['logits'].as_posix().replace(old_model_id, new_model_id))
        config['inference']['output_dir']['hidden'] = Path(config['inference']['output_dir']['hidden'].as_posix().replace(old_model_id, new_model_id))
        return config

    def run(self):
        for i in range(len(self.configures)):
            config = self.configures[i]
            last_config = self.configures[i - 1] if i >= 1 else defaultdict(dict)
            self.setup_torch_environments(config, last_config)
            self.create_data_engine(config, last_config)
            self.generate_periods(config, last_config)
            self.generate_universe(config, last_config)
            for repeat in config['num_repeats']:
                config = self.change_model_id(config, repeat)
                # 遍历nn_periods中的每个训练周期
                print('Train for every period in NN periods ...')
                for year, nn_period in self.nn_periods.items():
                    if nn_period['val'][1][-1] > self.data_engine.trade_dates[-1]:
                        continue
                    self.train_nn(year, nn_period, config, last_config)
                    for i in range(10):
                        torch.no_grad()
                        torch.cuda.empty_cache()
                    self.infer_nn(year, nn_period, config, last_config)
                    for i in range(10):
                        torch.no_grad()
                        torch.cuda.empty_cache()
                    self.last_nn_period = nn_period


class TaskTree:

    def __init__(self, configure_files: List[str]):
        yaml.add_constructor('!join', join)
        yaml.add_constructor('!join_path', join_path)
        self.configures = self.load_configures(configure_files)
        self.last_nn_period = None
    
    def load_configures(self, configure_files):
        configures = []
        for config_file in configure_files:
            with open(config_file, 'r', encoding='utf-8') as file:
                config = yaml.load(file, yaml.FullLoader)
            configures.append(config)
        return configures

    def generate_periods(self, config, last_config=defaultdict(dict)):
        # 获取训练周期（训练集/验证集/测试集）划分
        if config['calendar'] == last_config['calendar']:
            print('The same periods has been generated, skip this node.')
            return
        self.calendar = config['calendar']['class']()
        self.tree_periods = self.calendar.cut_periods(**config['calendar']['tree'])
        self.tree_simplified_periods = self.calendar.simplify_periods(self.tree_periods)

    def generate_universe(self, config, last_config=defaultdict(dict)):
        # 创建训练集/测试集股票池
        if config['universe'] == last_config['universe']:
            print('The same universes has been generated, skip this node.')
            return
        self.univ = config['universe']['class']()
        self.train_universe = self.univ(**config['universe']['train'])
        self.pred_universe = self.univ(**config['universe']['pred'])
        print(f'{datetime.now()}: Create train and predict universe.')
        print('\tTrain Universe: '  + '\n\t'.join(str(self.train_universe).split('\n')))
        print('\tPredict Universe: ' + '\n\t'.join(str(self.pred_universe).split('\n')))
    
    def create_model(self, config, last_config):
        if config['model'] == last_config['model']:
            print('The same model instance has been created, skip this node.')
            return
        self.model = config['model']['class']()

    def train(self, config):
        self.model.run(
            periods        = self.tree_periods,
            train_universe = self.train_universe,
            pred_universe  = self.pred_universe,
            **config['run']
        )

    def change_model_id(self, config, repeat):
        old_model_id = config['model']['model_id']
        new_model_id = f'model_{repeat}'
        config['model']['model_id'] = new_model_id
        for i, input_dir in enumerate(config['run']['saved_inputs_dirs']):
            config['run']['saved_inputs_dirs'][i] = Path(input_dir.as_posix().replace(old_model_id, new_model_id))
        config['run']['saved_outputs_dir'] = Path(config['run']['saved_outputs_dir'].as_posix().replace(old_model_id, new_model_id))
        return config

    def run(self):
        for i in range(len(self.configures)):
            config = self.configures[i]
            last_config = self.configures[i - 1] if i >= 1 else defaultdict(dict)
            for repeat in config['num_repeats']:
                print(repeat)
                config = self.change_model_id(config, repeat)
                self.generate_periods(config, last_config)
                self.generate_universe(config, last_config)
                self.create_model(config, last_config)
                self.train(config)


