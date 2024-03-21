'''
Author: WangXiang
Date: 2024-03-21 20:42:41
LastEditTime: 2024-03-21 20:48:10
'''

import numpy as np
import pandas as pd


def format_unstack_table(data: pd.DataFrame) -> pd.DataFrame:
    if np.issubdtype(data.columns.dtype, np.object_):
        data = data.loc[:, data.columns.str[0].str.isdigit()]
        data.columns = data.columns.str[:6].astype(int)
    if np.issubdtype(data.index.dtype, np.object_):
        data.index = data.index.astype(int)
    data.index.name, data.columns.name = 'trade_date', 'ticker'
    return data