'''
Author: WangXiang
Date: 2024-04-20 15:31:22
LastEditTime: 2024-04-20 15:31:28
'''

import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from .configure import TrainingArguments


@dataclass
class TrainerState:
    '''
    用于记录训练过程中的状态
    '''
    epoch: Optional[int] = field(
        default=0,
        metadata={'help': '当前训练到第几个epoch'}
    )
    step_in_epoch: Optional[int] = field(
        default=0,
        metadata={'help': '当前epoch中训练到第几个step'}
    )
    global_step: int = field(
        default=0,
        metadata={'help': '当前一共训练到第几个step（不考虑epoch）'}
    )
    best_metric: Dict[str, float] = field(
        default=None,
        metadata={'help': '保存当前最好的模型的metric'}
    )
    metrics: List[Dict[str, float]] = field(
        default=None,
        metadata={'help': '保存所有验证节点下的模型的metric'}
    )
    scheduler_history: List[float] = field(
        default=None,
        metadata={'help': '保存所有验证节点下的学习率'}
    )
    progress: Optional[Any] = field(
        default=None,
        metadata={'help': '保存当前模型的其他状态，详见trainer.py中的Trainer'}
    )

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = []
        if self.scheduler_history is None:
            self.scheduler_history = []

    def save_to_json(self, json_path):
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + '\n'
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_string)
        f.close()

    @classmethod
    def load_from_json(cls, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            text = f.read()
        f.close()
        return cls(**json.loads(text))


@dataclass
class TrainerControl:
    '''
    神经网络训练器控制逻辑
    '''

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        # 训练开始时的逻辑
        self.should_training_stop = False

    def _new_epoch(self):
        # epoch开始时的逻辑
        self.should_epoch_stop = False

    def _new_step(self):
        # step开始时的逻辑，step是指一次前向传播和梯度回传
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


class TrainerCallback:
    '''
    神经网络训练器回调函数和回调逻辑
    '''

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass


class CallbackHandler(TrainerCallback):

    def __init__(self, callbacks, model, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            print(
                f'You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current list of callbacks is\n:' + self.callback_list)
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return '\n'.join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_init_end', args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_training_stop = False
        return self.call_event('on_train_begin', args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_train_end', args, state, control)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        return self.call_event('on_epoch_begin', args, state, control)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_epoch_end', args, state, control)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event('on_step_begin', args, state, control)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_substep_end', args, state, control)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_step_end', args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_evaluate = False
        return self.call_event('on_evaluate', args, state, control)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save = False
        return self.call_event('on_save', args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        control.should_log = False
        return self.call_event('on_log', args, state, control, logs=logs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event('on_prediction_step', args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            if result is not None:
                control = result
        return control


class DefaultFlowCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 若eval_step大于0，则在每个step结束时判断当前step是否需要做验证，若需要，则should_evaluate为True
        if args.eval_step > 0:
            if state.global_step % args.eval_step == 0:
                control.should_evaluate = True

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 若eval_step等于0，则在每个epoch结束时进行验证
        if args.eval_step == 0:
            control.should_evaluate = True


class ProgressCallback(TrainerCallback):

    names = ['Epoch', 'Batch', 'Train Loss', 'Val Loss', 'Elapsed']
    bins = [7, 7, 44, 44, 9]

    def __init__(self):
        self.total_width = sum(self.bins) + len(self.bins) + 1
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_log = False
        if state.step_in_epoch % args.log_step == 0:
            self.log_row(args, state)
            control.should_log = True
        # 用于每个step都要更新学习率的scheduler
        lr_scheduler = kwargs.get('lr_scheduler')
        # lr_scheduler.step()
        return control

    def on_epoch_begin(self, args:TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print('-' * self.total_width)
        head = ' '
        for i in range(len(self.names)):
            name = self.names[i]
            bin = self.bins[i]
            head += f'{name:^{bin}}'
            if i < len(self.names) - 1:
                head += '|'
        print(head)
        print('-' * self.total_width)
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not (args.eval_step == 0 and not control.should_evaluate):
            print('-' * self.total_width)
            self.log_row(args, state)
            print('')
            print('')

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print('-' * self.total_width)
        eval_metric, eval_loss = self.log_row(args, state)
        print('-' * self.total_width)
        state.metrics.append(eval_metric)
        lr_scheduler = kwargs.get('lr_scheduler')
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            cur_lr = lr_scheduler.__dict__.get('_last_lr', [0])[0]
            lr_scheduler.step(eval_loss)
            new_lr = lr_scheduler.__dict__.get('_last_lr', [0])[0]
            if new_lr < cur_lr and args.trace_back:
                control.should_trace_back = True
            else:
                control.should_trace_back = False
            state.scheduler_history.append(lr_scheduler.__dict__.get('_last_lr', [0])[0])
            print(f'Current learning rate = {state.scheduler_history[-1]}')
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print('Training completed.\n\n')
    
    def log_row(self, args: TrainingArguments, state: TrainerState):
        eval_metric, eval_loss = None, None
        row = ' '
        for i in range(len(self.names)):
            name = self.names[i]
            bin = self.bins[i]
            if name == 'Train Loss' or name == 'Val Loss':
                text, metric, loss = self.parse_loss(state.progress[name], args)
                if name == 'Val Loss':
                    eval_metric, eval_loss = metric, loss
            else:
                text = state.progress[name]
            if isinstance(text, float):
                text = f'{text:.2f}'
            row += f'{text:^{bin}}'
            if i < len(self.names) - 1:
                row += '|'
        print(row)
        return eval_metric, eval_loss

    def parse_loss(self, content, args):
        if content is None or content == {}:
            return '-', None, None
        loss = np.average(content['loss'], weights=content['sample_counts'])
        ic = np.average(content['-ic'], weights=content['sample_counts'])
        corr = np.average(content['corr'], weights=content['sample_counts'])
        text = f'{loss:.6f} = {ic:.6f}(-ic) + {corr:.6f}(corr)'
        metric = {'loss': loss, '-ic': ic, 'corr': corr}
        return text, metric, metric[args.metric_name]


class EarlyStoppingCallback(TrainerCallback):

    def __init__(self, patience: int = 40, threshold: Optional[float] = 0.0):
        # patiance: 至多几次验证集loss不下降，则提前停止
        # threshold: 判定是否下降，即下降幅度至少要达到多少才算下降
        self.patience = patience
        self.threshold = threshold
        self.patience_counter = 0
    
    def check_metric_value(self, args, best_metric_value, metric_value):
        operator = np.greater if args.greater_is_better else np.less
        if best_metric_value is None or (
            operator(metric_value, best_metric_value) and abs(metric_value - best_metric_value) > self.threshold
        ):
            self.patience_counter = 0
            return 1
        else:
            self.patience_counter += 1
            return 0
    
    def on_evaluate(self, args, state, control, **kwargs):
        if not state.best_metric:
            best_metric_value = None
        else:
            best_metric_value = state.best_metric[args.metric_name]
        metric_value = state.metrics[-1][args.metric_name]
        check_result = self.check_metric_value(args, best_metric_value, metric_value)
        if check_result:
            state.best_metric = state.metrics[-1]
            control.should_save = True
        if self.patience_counter >= self.patience:
            control.should_training_stop = True
        return control