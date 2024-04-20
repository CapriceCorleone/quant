'''
Author: WangXiang
Date: 2024-04-19 22:45:18
LastEditTime: 2024-04-19 22:50:15
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICLoss(nn.Module):

    def __init__(self, dim: int = 0, eps: float = 1e-8, coef: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.coef = coef

    def calc_ic(self, x1, x2):
        return F.cosine_similarity(x1 - x1.mean(), x2 - x2.mean(), self.dim, self.eps)
    
    def calc_corr(self, h):
        cov = h.T @ h
        std = (h ** 2).sum(dim=0, keepdim=True) ** (1 / 2)
        corr = cov / (std.T @ std + self.eps)
        corr = torch.nan_to_num(corr)
        n = (corr.size()[0] - 1) * corr.size()[0] / 2
        return (corr.triu(1) ** 2).sum() / n
    
    def forward(self, labels, logits, hidden = None, **kwargs):
        if labels is None or logits is None:
            return None, None
        if len(labels.squeeze()) <= 100:
            return None, None
        L1 = - self.calc_ic(labels.squeeze(), logits.squeeze())
        L2 = self.calc_corr(hidden) * self.coef
        L = L1 + L2
        metric = {
            'loss': L.detach().item(),
            '-ic': L1.detach().item(),
            'corr': L2.detach().item()
        }
        return L, metric