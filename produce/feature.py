'''
Author: WangXiang
Date: 2024-04-14 21:07:55
LastEditTime: 2024-04-14 21:14:39
'''

import os
os.chdir('E:/quant/produce/')
import sys
sys.path.append('E:/')
import warnings
warnings.filterwarnings('ignore')


from cxml import conf
from cxml.feature.daily_bar import DailyBarFeatureManager
from cxml.feature.minute_bar import MinuteBarFeatureManager
from cxml.feature import formulas


if __name__ == "__main__":
    
    """ daily bar & weekly bar """
    dbf = DailyBarFeatureManager(
        feature_list = ['daily_bar', 'weekly_bar'],
        formulas     = [formulas],
        feature_dir  = conf.PATH_DATA_DAILY_FEATURE
    )
    dbf.run()

    """ risk factor """
    dbf = DailyBarFeatureManager(
        feature_list = ['risk_factor'],
        formulas     = [formulas],
        feature_dir  = conf.PATH_DATA_RISK_FACTOR
    )
    dbf.run()

    """ minute bar features """
    mbf = MinuteBarFeatureManager(
        feature_list  = ['long_short_game'],
        frequency     = '1m',
        num_processes = 8,
        formualas     = [formulas],
        feature_dir   = conf.PATH_DATA_MINUTE_FEATURE
    )
    mbf.run(20151231)
