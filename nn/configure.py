'''
Author: WangXiang
Date: 2024-04-20 15:28:21
LastEditTime: 2024-04-20 15:28:25
'''

import os
import json
import torch
from enum import Enum
from dataclasses import dataclass, asdict, field


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={'help': '训练模型的保存文件夹'}
    )
    device: str | torch.device = field(
        metadata={'help': '显卡'}
    )
    batch_type: str = field(
        default='date',
        metadata={'help': 'batching类型'}
    )
    train_batch_size: int = field(
        default=1,
        metadata={
            'help': '训练集的batch_size，对于损失函数为 IC 的时序选股模型，batch_size指的是每次输入的截面数量，其中每个截面包括当日所有的股票数据'}
    )
    train_shuffle: bool = field(
        default=True,
        metadata={'help': '训练集数据在训练时是否打乱顺序'}
    )
    eval_batch_size: int = field(
        default=1,
        metadata={'help': '验证集的batch_size'}
    )
    num_train_epochs: int = field(
        default=100,
        metadata={
            'help': '单次训练的epoch上限，对于epoch的定义详见trainer_callback.py中的ProgressCallback和trainer.py中的Trainer'}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={'help': '初始学习率'}
    )
    eval_step: int = field(
        default=0,
        metadata={'help': '模型验证的频率，按step计算，step指的是梯度回传的次数（这里没有使用多次前向传播后将梯度累加再回传，都是一次前向传播，一次梯度回传）'
                          '若eval_step为0，则仅当每个epoch结束时才会进行一次验证'}
    )
    log_step: int = field(
        default=10,
        metadata={'help': '控制台输出模型当前进度和其他指标的频率，按step计算'}
    )
    max_grad_value: float = field(
        default=5.0,
        metadata={'help': '梯度截断，梯度每个维度的绝对值的最大值，对应函数torch.nn.utils.clip_grad_value_'}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={'help': '梯度截断，梯度模长的最大值，对应函数torch.nn.utils.clip_grad_norm_'}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={'help': '损失函数越大越好还是越小越好，True对应损失越大越好，False对应损失越小越好'}
    )
    metric_name: str = field(
        default='-ic',
        metadata={
            'help': '损失函数值在metric中对应的key值，metric是一个字典，用于临时存储包括loss在内的对于模型评价的各种中间状态'}
    )
    seed: int = field(
        default=0,
        metadata={'help': '随机种子'}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={'help': 'optimizer中的weight_decay'}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={'help': 'Adam优化器中的betas中的第一个值'}
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={'help': 'Adam优化器中的betas中的第二个值'}
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={'help': 'Adam优化器中的eps'}
    )
    trace_back: bool = field(
        default=False,
        metadata={'help': '学习率调整后是否将模型回溯至上一个最优节点'}
    )
    train_prefetch_factor: int = field(
        default=2,
        metadata={'help': '训练集DataLoader的prefetch_factor参数'}
    )
    train_num_workers: int = field(
        default=0,
        metadata={'help': '训练集DataLoader的num_workers参数'}
    )
    train_persistent_workers: bool = field(
        default=False,
        metadata={'help': '训练集DataLoader的persistent_workers参数'}
    )
    train_use_thread_workers: bool = field(
        default=False,
        metadata={'help': '训练集DataLoader的use_thread_workers参数 (ThreadDataLoader)'}
    )
    train_buffer_size: int = field(
        default=1,
        metadata={'help': '训练集DataLoader的buffer_size参数'}
    )

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        self.eval_shuffle = False

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f'{k}={v},\n' for k, v in sorted(self_as_dict.items())]
        attrs_as_str = '\t'.join(attrs_as_str)
        return f"{self.__class__.__name__}(\n\t{attrs_as_str})"

    __repr__ = __str__

    def to_dict(self):
        d = asdict(self)
        for k, v in d.itmes():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self):
        d = self.to_dict()
        valid_types = [bool, int, float, str, torch.Tensor]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}