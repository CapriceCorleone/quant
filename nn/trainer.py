'''
Author: WangXiang
Date: 2024-04-20 15:32:13
LastEditTime: 2024-04-20 15:32:24
'''

import os
import re
import time
import yaml
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Union, Optional, Callable, Tuple, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_utils import PreTrainedModel

from .dataloader import ThreadDataLoader
from .models import DatasetProcessor
from .configure import TrainingArguments
from .callback import (
    TrainerState,
    TrainerControl,
    TrainerCallback,
    CallbackHandler,
    DefaultFlowCallback,
    ProgressCallback,
    EarlyStoppingCallback,
)


DEFAULT_CALLBACKS = [DefaultFlowCallback, ProgressCallback, EarlyStoppingCallback]
WEIGHTS_NAME = 'pytorch_model.bin'
TRAINING_ARGS_NAME = 'training_args.bin'
TRAINER_STATE_NAME = 'trainer_state.json'
OPTIMIZER_NAME = 'optimizer.pt'
SCHEDULER_NAME = 'scheduler.pt'
_re_checkpoint = re.compile(r'^checkpoint\-(\d+)\-(\d+)$')


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    return model


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: (
        int(_re_checkpoint.search(x).groups()[0]),
        int(_re_checkpoint.search(x).groups()[1])
    )))


class Trainer:

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        args: TrainingArguments,
        train_dataset: Dataset,
        dataset_processor: DatasetProcessor,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        loss_fn: nn.Module = None,
    ):
        self.args = args
        self.train_dataset = train_dataset
        self.dataset_processor = dataset_processor
        self.eval_dataset = eval_dataset
        self.model = model.to(args.device)
        self.optimizer, self.lr_scheduler = optimizers
        self.loss_fn = loss_fn
        callbacks = DEFAULT_CALLBACKS if callbacks is None else callbacks
        self.callback_handler = CallbackHandler(callbacks, self.model, self.optimizer, self.lr_scheduler)
        self.state = TrainerState()
        self.control = TrainerControl()
    
    def add_callback(self, callback):
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        return self.callback_handler.pop_callback(callback)
    
    def remove_callback(self, callback):
        self.callback_handler.remove_callback(callback)

    # def get_collate_fn(self, batch_type):  # to be deprecated
    #     if batch_type == 'date':
    #         return lambda x: x[0]
    #     return
    
    # def get_train_dataloader(self):  # use torch.utils.data.DataLoader
    #     train_dataset = self.train_dataset
    #     # collate_fn = self.get_collate_fn(self.args.batch_type)
    #     return DataLoader(
    #         train_dataset,
    #         batch_size         = self.args.train_batch_size,
    #         shuffle            = self.args.train_shuffle,
    #         prefetch_factor    = self.args.train_prefetch_factor,
    #         num_workers        = self.args.train_num_workers,
    #         persistent_workers = self.args.train_persistent_workers,
    #     )
        
    def get_train_dataloader(self):  # use monai.data.ThreadDataLoader
        train_dataset = self.train_dataset
        return ThreadDataLoader(
            train_dataset,
            buffer_size        = self.args.train_buffer_size,
            use_thread_workers = self.args.train_use_thread_workers,
            batch_size         = self.args.train_batch_size,
            shuffle            = self.args.train_shuffle,
            prefetch_factor    = self.args.train_prefetch_factor,
            num_workers        = self.args.train_num_workers,
            persistent_workers = self.args.train_persistent_workers,
        )
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        # collate_fn = self.get_collate_fn(self.args.batch_type)
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=self.args.eval_shuffle,
        )
    
    def get_test_dataloader(self, test_dataset: Dataset):
        # collate_fn = self.get_collate_fn(self.args.batch_type)
        return DataLoader(
            test_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=self.args.eval_shuffle,
        )

    def train(self, **kwargs):
        '''
        训练流程的基本顺序是
        1. 训练开始
        2. epoch开始
        3. step开始
        4. step结束，判断是否需要evaluate
          （若需要）进行evaluate，判断是否需要early stop
            （若需要）终止训练
        5. epoch结束，判断是否需要evaluate和early stop
          （若需要）进行evaluate，判断是否需要early stop
            （若需要）终止训练
        6. 训练结束
        '''
        args = self.args
        config = kwargs.get('config', {})

        # 创建模型存储目录
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

        train_dataloader = self.get_train_dataloader()
        model = self.model
        print(f'Number of samples per Epoch = {len(self.train_dataset)}')
        print(f'Train batch size = {args.train_batch_size}')
        print(f'Number of batches per Epoch = {len(train_dataloader)}')

        # state中的计数器进行初始化
        self.state.epoch = 0
        self.state.num_train_epochs = args.num_train_epochs

        # callback_handler中的模型组件进行初始化载入
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        # 清空模型梯度
        model.zero_grad()

        # 执行训练开始之前（on_train_begin）的callbacks
        self.control  = self.callback_handler.on_train_begin(args, self.state, self.control)

        # 开始训练
        for epoch in range(args.num_train_epochs):

            self.state.epoch += 1
            epoch_iterator = train_dataloader
            steps_in_epoch = len(epoch_iterator)

            # 执行epoch开始之前（on_epoch_begin）的callbacks
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            
            epoch_metric = {}  # 存储本次【epoch】周期的metric
            train_metric = {}  # 存储本次【训练-验证】周期的训练集的metric（注：【训练-验证】周期和epoch周期不相同）
            log_metric   = {}  # 存储本次【log】周期的metric
            
            epoch_start_time      = time.time()  # 本次【epoch】周期的开始时间
            mini_epoch_start_time = time.time()  # 本次【训练-验证】周期的开始时间
            start_time            = time.time()  # 本次【log】周期的开始时间

            # 本次【epoch】周期开始训练
            for step, inputs in enumerate(epoch_iterator):

                for i in range(5): torch.cuda.empty_cache()  # 清空CUDA缓存

                # 执行step开始之前（on_step_begin）的callbacks
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                self.state.step_in_epoch = step + 1
                self.state.global_step += 1
                
                # 一个step通过模型，并计算得到metric（step可以理解为batch）
                step_metric = self.training_step(model, inputs)
                
                # 梯度截断
                if args.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                elif args.max_grad_value is not None:
                    nn.utils.clip_grad_value_(model.parameters(), args.max_grad_value)
                
                # 梯度回传
                self.optimizer.step()

                # 更新学习率
                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    print(self.optimizer.__dict__)
                    self.lr_scheduler.step()
                    self.state.scheduler_history.append(self.lr_scheduler.__dict__.get('_last_lr', [0])[0])
                    print(f'Current learning rate = {self.state.scheduler_history[-1]}')

                # 清空模型梯度
                model.zero_grad()

                # 将当前step的metric更新到epoch_metric、train_metric和log_metric中
                epoch_metric = self.update_metric(epoch_metric, step_metric)
                train_metric = self.update_metric(train_metric, step_metric)
                log_metric   = self.update_metric(log_metric, step_metric)

                # 将log_metric更新到state.progress（训练状态缓存器）中
                self.state.progress = self.update_progress(log_metric, None, start_time=start_time)

                # 执行step结束之后（on_step_end）的callbacks
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                # 若当前step结束后进行了log，则清空log_metric的缓存，并重置【log】周期的开始时间
                if self.control.should_log:
                    log_metric = {}
                    start_time = time.time()

                # 判断当前step结束后是否需要验证
                if self.control.should_evaluate:
                    # 进行验证
                    eval_metric = self.evaluate()
                    # 在state.progress中更新训练集metric和验证集metric
                    self.state.progress = self.update_progress(train_metric, eval_metric, start_time=mini_epoch_start_time)
                    # 执行验证结束后（on_evaluate）的callbacks
                    self.control = self.callback_handler.on_evaluate(args, self.state, self.control)
                    # 清空log_metric和train_metric的缓存
                    log_metric, train_metric = {}, {}
                    # 判断是否需要保存模型
                    if self.control.should_save:
                        self.save_checkpoint(config)  # 保存模型
                    # 判断是否需要回溯模型
                    if self.control.should_trace_back:
                        self.trace_back()  # 回溯模型
                    # 重置【log】周期和【训练-验证】周期的起始时间
                    start_time            = time.time()
                    mini_epoch_start_time = time.time()
                
                # 判断当前step结束后是否需要停止本次【epoch】周期或整个训练流程
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            
            # 本次【epoch】周期结束后，更新state.progress
            self.state.progress = self.update_progress(epoch_metric, None, start_time=epoch_start_time, steps_in_epoch=steps_in_epoch)

            # 执行epoch结束后（on_epoch_end）的callbacks
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            # 判断当前epoch结束后是否需要验证（验证流程与前面相同）
            if self.control.should_evaluate:
                eval_metric = self.evaluate()
                self.state.progress = self.update_progress(epoch_metric, eval_metric, start_time=epoch_start_time)
                self.control = self.callback_handler.on_evaluate(args, self.state, self.control)
                if self.control.should_save:
                    self.save_checkpoint(config)
                if self.control.should_trace_back:
                    self.trace_back()

            # 判断是否需要停止整个训练流程
            if self.control.should_training_stop:
                break
        
        # 执行训练结束后（on_train_end）的callbacks
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        del epoch_iterator
        del train_dataloader

    def get_sample_in_inputs(self, inputs):
        # 计算每个batch数据中实际的样本数量，对以 IC 为损失函数的时序选股模型来说，一个截面就是一个样本，其中一个截面包含当日所有股票的输入数据
        if isinstance(inputs, Mapping):
            return self.get_sample_in_inputs(inputs[list(inputs.keys())[0]])
        elif isinstance(inputs, (tuple, list)):
            return self.get_sample_in_inputs(inputs[0])
        elif isinstance(inputs, torch.Tensor):
            return inputs.size()[0]
        return 1

    def update_metric(self, old, new):
        # 更新模型metric，old为原先metric，new为新的metric，若old中有值则更新，否则直接用new替换
        if old:
            if new:
                for k in old.keys():
                    old[k].append(new[k])
        else:
            old = {k: [v] for k, v in new.items()}
        return old

    def update_progress(self, train_metric, eval_metric, **kwargs):
        # 更新模型进度，包括epoch，batch（step），训练集metric和验证集metric（没有则为空）
        epoch = self.state.epoch
        batch = self.state.step_in_epoch
        steps_in_epoch = kwargs.get('steps_in_epoch', -1)
        batch = '-' if batch == steps_in_epoch or self.control.should_evaluate else batch
        start_time = kwargs.get('start_time', None)
        elapsed = '-' if start_time is None else time.time() - start_time
        progress = {'Epoch': epoch, 'Batch': batch, 'Train Loss': train_metric, 'Val Loss': eval_metric, 'Elapsed': elapsed}
        return progress

    def training_step(self, model, inputs):
        # 训练逻辑，一次前向传播 + 梯度回传
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss, metric = self.compute_loss(model, inputs)
        if loss is not None:
            loss.backward()
            metric['sample_counts'] = self.get_sample_in_inputs(inputs)
            return metric
        return {}
    
    def compute_loss(self, model, inputs):
        # 计算损失函数
        outputs = model(inputs)
        loss, metric = self.loss_fn(**outputs)
        return loss, metric
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        # 验证流程
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_metric = {}
        model = self.model
        model.eval()
        for step, inputs in enumerate(eval_dataloader):
            for i in range(5): torch.cuda.empty_cache()  # 清空CUDA缓存
            step_metric = self.evaluation_step(model, inputs)
            eval_metric = self.update_metric(eval_metric, step_metric)
        del eval_dataloader
        return eval_metric

    def evaluation_step(self, model, inputs):
        # 验证逻辑，与训练逻辑类似，但不需要梯度回传
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, metric = self.compute_loss(model, inputs)
        if loss is not None:
            metric['sample_counts'] = self.get_sample_in_inputs(inputs)
            return metric
        return {}
    
    def predict(self, test_dataset: Dataset):
        # 预测流程
        test_dataloader = self.get_test_dataloader(test_dataset)
        model = self.model
        model.eval()
        result_logits = []
        result_hidden = []
        for step, inputs in enumerate(tqdm(test_dataloader, ncols=80, desc='get hidden')):
            output = self.prediction_step(model, inputs)
            if output['ticker'] is None:
                continue
            logits_df = pd.DataFrame({
                'trade_date': output['trade_date'].detach().cpu().numpy().astype(np.float64).astype(int),
                'ticker': output['ticker'].detach().cpu().numpy().astype(np.float64).astype(int),
                'logits': output['hidden'].mean(dim=1).squeeze().detach().cpu().numpy().astype(np.float64),
            })
            result_logits.append(logits_df)
            if output.get('hidden', None) is not None:
                hidden_df = pd.DataFrame({
                    'trade_date': output['trade_date'].detach().cpu().numpy().astype(np.float64).astype(int),
                    'ticker': output['ticker'].detach().cpu().numpy().astype(np.float64).astype(int),
                    'labels': output['labels'].squeeze().detach().cpu().numpy().astype(np.float64),
                })
                num_feats = output['hidden'].shape[1]
                hidden_df[[f'feature_{i}' for i in range(num_feats)]] = output['hidden'].detach().cpu().numpy().astype(np.float64)
                result_hidden.append(hidden_df)
            del output
            torch.cuda.empty_cache()
        result_logits = pd.concat(result_logits, axis=0, ignore_index=True)
        if len(result_hidden) > 0:
            result_hidden = pd.concat(result_hidden, axis=0, ignore_index=True)
        del test_dataloader
        return result_logits, result_hidden
    
    def prediction_step(self, model, inputs):
        # 预测逻辑，与验证逻辑类似，但是需要将预测结果保存到字典中
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(inputs)
        return outputs

    def save_checkpoint(self, config):
        # 保存模型权重、参数和状态等
        checkpoint_folder = f'checkpoint-{self.state.epoch}-{self.state.global_step}'
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        if os.path.exists(output_dir):
            print(f'Overwriting checkpoint in {output_dir}.')
            for fn in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, fn))
            os.rmdir(output_dir)
        self.save_model(output_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        rng_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'cpu': torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states['cuda'] = torch.cuda.random.get_rng_state()
        torch.save(rng_states, os.path.join(output_dir, 'rng_state.pth'))
        with open(os.path.join(output_dir, 'task_configure.yaml'), 'w') as f:
            yaml.dump(config, f)

    def save_model(self, output_dir: Optional[str] = None, state_dict=None):
        # 保存模型权重
        output_dir = self.args.output_dir if output_dir is None else output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.config._name_or_path = output_dir
        print(f'Saving model checkpoint to {output_dir}.')
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
            else:
                print('Trainer.model is not a `PreTrainedModel`, only saving its state dict.')
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        
    def trace_back(self):
        checkpoint = get_last_checkpoint(self.args.output_dir)
        print('Trace back to model', checkpoint)
        state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME), map_location='cpu')
        self._load_state_dict_in_model(state_dict)
        self._load_optimizer(checkpoint)
        self.state = TrainerState.load_from_json(os.path.join(checkpoint, TRAINER_STATE_NAME))
        print(self.lr_scheduler.__dict__.get('_last_lr', [0])[0])

    def set_seed(self, seed: int = 0):
        # 设置随机种子
        seed = seed if self.args.seed is None else self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.load_state_dict(state_dict, strict=False)
        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(self.model._keys_to_ignore_on_save):
                self.model.tie_weights()
            else:
                print(f'Warning: There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.')
        if len(load_result.unexpected_keys) != 0:
            print(f'Warning: There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.')
    
    def _load_optimizer(self, checkpoint):
        if checkpoint is None:
            return
        if os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME)):
            self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location='cpu'))
            for i in range(len(self.optimizer.param_groups)):
                self.optimizer.param_groups[i]['lr'] = self.lr_scheduler.__dict__.get('_last_lr', [0])[0]
    
    def _load_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        if os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME)):
            map_location = self.args.device
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME), map_location=map_location))
    
    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return 
        rng_file = os.path.join(checkpoint, 'rng_state.pth')
        if not os.path.exists(rng_file):
            print('Did not find an RNG file.')
            return
        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state['python'])
        np.random.set_state(checkpoint_rng_state['numpy'])
        torch.random.set_rng_state(checkpoint_rng_state['cpu'])
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state(checkpoint_rng_state['cuda'])
    
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, np.ndarray):
            return self._prepare_input(torch.as_tensor(data))
        elif isinstance(data, torch.Tensor):
            return data.to(self.args.device)
        return data
    
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        inputs = self._prepare_input(inputs)
        inputs = self.dataset_processor(inputs[0])
        return inputs