'''
Author: WangXiang
Date: 2024-04-20 15:45:18
LastEditTime: 2024-04-20 15:53:41
'''

import os
os.chdir('E:/quant/unittest/')

import sys
sys.path.append('E:/')

from .. import conf
from ..nn import TaskNN, TaskTree, SimilarLabelManager
from ..feature import DailyBarFeatureManager, formulas


# %%
if __name__ == "__main__":

    # feature update
    feature_manager = DailyBarFeatureManager(
        feature_list = ['daily_bar', 'weekly_bar'],
        formulas     = [formulas],
        feature_dir  = conf.PATH_DATA_DAILY_FEATURE
    )
    feature_manager.run(20050104)

    risk_factor_manager = DailyBarFeatureManager(
        feature_list = ['risk_factor'],
        formulas     = [formulas],
        feature_dir  = conf.PATH_DATA_RISK_FACTOR
    )
    risk_factor_manager.run(20050104)

    # label update
    lm = SimilarLabelManager(window=10, similar_window=243, skip=1)
    lm(20050104)

    # train nn
    configure_files = ['gru.yaml']
    task = TaskNN(configure_files)
    task.run()

    # train lightgbm
    configure_files = ['gru_lgb.yaml']
    task = TaskTree(configure_files)
    task.run()