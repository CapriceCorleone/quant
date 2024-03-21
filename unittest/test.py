'''
Author: WangXiang
Date: 2024-03-21 20:25:56
LastEditTime: 2024-03-21 21:10:29
'''

import os
os.chdir('E:/quant/')

import sys
sys.path.append('E:/')


if __name__ == "__main__":
    
    # DataLoader
    from quant.core import DataLoader
    dl = DataLoader()
    AShareCalendar = dl.load('AShareCalendar')

    # DataMaintainer
    #  full time: 20-40 seconds
    #  increment time: 3-5 seconds
    from quant.core import DataMaintainer
    dm = DataMaintainer()
    dm.update_index()
    dm.update_stock_description()
    dm.update_stock_quote()
    dm.update_stock_size()
    
    # Universe
    #  full time: 3-5 seconds
    from quant.core import Universe
    univ = Universe()
    market_univ = univ(listed_days=120, continuous_trade_days=20, include_st=False, include_suspend=False, include_price_limit=False)
    hs300_univ = univ(listed_days=120, continuous_trade_days=20, include_st=False, include_suspend=False, include_price_limit=False, index_code='000300.SH')


