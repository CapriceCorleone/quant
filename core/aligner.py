'''
Author: WangXiang
Date: 2024-03-21 20:36:55
LastEditTime: 2024-03-21 20:54:28
'''

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from .. import conf


class Aligner:

    index_path = conf.PATH_DATA_BASIC_PROCESSED / 'index.npy'

    def __init__(self) -> None:
        self.load_index()

    def load_index(self) -> None:
        index = np.load(self.index_path, allow_pickle=True).item()
        self.trade_dates = index['trade_date']
        self.tickers = index['ticker']
    
    def align(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.reindex(index=self.trade_dates, columns=self.tickers)
    
    def append(self, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        new = self.align(new)
        old = old[old.index < new.index.min()]
        data = pd.concat([old, new], axis=0)
        return self.align(data)
    
    def __str__(self) -> str:
        return f"""
        Universal Aligner
            Trade dates : {self.trade_dates[0]} - {self.trade_dates[-1]}, total num = {len(self.trade_dates)}
            Tickers     : {self.tickers[0]} - {self.tickers[-1]}, total num = {len(self.tickers)}
        """