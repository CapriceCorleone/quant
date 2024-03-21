'''
Author: WangXiang
Date: 2024-03-21 20:25:56
LastEditTime: 2024-03-21 22:08:56
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
    dm.update_index_quote()
    
    # Universe
    #  full time: 3-5 seconds
    from quant.core import Universe
    univ = Universe()
    market_univ = univ(listed_days=120, continuous_trade_days=20, include_st=False, include_suspend=False, include_price_limit=False)
    hs300_univ = univ(listed_days=120, continuous_trade_days=20, include_st=False, include_suspend=False, include_price_limit=False, index_code='000300.SH')
    
    AShareIndustriesClassCITICS = dl.load('AShareIndustriesClassCITICS')
    info = AShareIndustriesClassCITICS[AShareIndustriesClassCITICS['INDUSTRIESNAME'] == '食品饮料'][['S_INFO_WINDCODE', 'ENTRY_DT', 'REMOVE_DT']]
    df = univ._format_universe(univ.arrange_info_table(info))

    # Calendar
    from quant.core import Calendar
    calendar = Calendar()
    print(calendar.to_frame())
    month_ends = calendar.month_ends
    

    