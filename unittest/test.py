'''
Author: WangXiang
Date: 2024-03-21 20:25:56
LastEditTime: 2024-03-23 14:51:50
'''

import os
os.chdir('E:/quant/')

import sys
sys.path.append('E:/')


from quant.core import DataLoader, DataMaintainer, Universe, Calendar
from quant.factor_test import FactorTester


if __name__ == "__main__":
    
    # DataLoader
    dl = DataLoader()
    AShareCalendar = dl.load('AShareCalendar')

    # DataMaintainer
    #  full time: 20-40 seconds
    #  increment time: 3-5 seconds
    dm = DataMaintainer()
    dm.update_index()
    dm.update_stock_description()
    dm.update_stock_quote()
    dm.update_stock_size()
    dm.update_index_quote()
    
    # Universe
    #  full time: 3-5 seconds
    univ = Universe()
    market_univ = univ(listed_days=120, continuous_trade_days=20, include_st=False, include_suspend=False, include_price_limit=False)
    hs300_univ = univ(listed_days=120, continuous_trade_days=20, include_st=False, include_suspend=False, include_price_limit=False, index_code='000300.SH')
    
    AShareIndustriesClassCITICS = dl.load('AShareIndustriesClassCITICS')
    info = AShareIndustriesClassCITICS[AShareIndustriesClassCITICS['INDUSTRIESNAME'] == '食品饮料'][['S_INFO_WINDCODE', 'ENTRY_DT', 'REMOVE_DT']]
    df = univ._format_universe(univ.arrange_info_table(info))

    # Calendar
    calendar = Calendar()
    print(calendar.to_frame())
    month_ends = calendar.month_ends
    

    # Factor Tester
    universe = univ(listed_days=120, continuous_trade_days=20, include_st=False, include_suspend=False, include_price_limit=False)
    ft = FactorTester(universe, 'M', 20161230, 20231229, 'vwap')