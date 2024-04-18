'''
Author: WangXiang
Date: 2024-03-30 14:03:56
LastEditTime: 2024-04-14 21:10:47
'''

import os
os.chdir('E:/quant/produce/')
import sys
sys.path.append('E:/')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from quant.core import conf
from quant.risk_model import RiskModelManager


# %%
def get_init_date(roll_back=20):
    file = conf.PATH_RISK_MODEL_DATA / 'model/factor_exposure.pkl'
    if file.exists():
        data = pd.read_pickle(file)
        index = np.sort(data[list(data.keys())[0]].index.values)
        init_date = index[-roll_back] if roll_back <= len(index) else index[0]
    else:
        init_date = 20081231
    return init_date


# %%
if __name__ == "__main__":
    init_date = get_init_date()
    rmm = RiskModelManager(conf.PATH_RISK_MODEL / 'structure.yaml', init_date=20081231)
    # factor = rmm.calc_risk_subfactor(rmm.structure[10]['subfactors'][0])
    factor, subfactor = rmm.calc_risk_factor(rmm.structure[10])
    rmm.calc_exposure()
