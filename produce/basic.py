'''
Author: WangXiang
Date: 2024-04-13 15:47:12
LastEditTime: 2024-04-13 15:47:57
'''

import os
os.chdir('E:/quant/')

import sys
sys.path.append('E:/')

import pandas as pd


from quant.core import Aligner, DataLoader, DataMaintainer, DataProcessor, AShareConsensus, FamaFrench3Factor


# %%
if __name__ == "__main__":

    dm = DataMaintainer()
    dm.update_index()
    dm.update_stock_description()
    dm.update_stock_quote(20050101)
    dm.update_stock_size(20050101)
    dm.update_index_quote(20050101)

    dp = DataProcessor([AShareConsensus, FamaFrench3Factor])
    dp.run()