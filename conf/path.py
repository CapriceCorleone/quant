'''
Author: WangXiang
Date: 2024-03-20 22:11:57
LastEditTime: 2024-03-20 22:32:51
'''

from pathlib import Path

PATH_ROOT = Path('E:/quant/')
PATH_DATA = PATH_ROOT / 'data'
PATH_DATA_BASIC = PATH_DATA / 'basic'  # 基础数据

PATH_FACTOR_TEST = PATH_ROOT / 'factor_test'  # 因子回测框架
PATH_PORTFOLIO_TEST = PATH_ROOT / 'portfolio_test'  # 组合回测框架
PATH_RISK_MODEL = PATH_ROOT / 'risk_model'  # 风险模型框架

PATH_PRODUCT = PATH_ROOT / 'product'  # 生产框架
PATH_FACTOR = PATH_PRODUCT / 'factor'  # 因子生产框架
PATH_MODEL = PATH_PRODUCT / 'model'  # 模型生产框架