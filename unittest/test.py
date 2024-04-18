'''
Author: WangXiang
Date: 2024-03-21 20:25:56
LastEditTime: 2024-04-18 21:32:28
'''

import os
os.chdir('E:/quant/')

import sys
sys.path.append('E:/')

import pandas as pd


from quant.core import (
    Aligner, DataLoader, DataMaintainer, Universe, Calendar, DataProcessor,
    format_unstack_table, winsorize_mad, stdd_zscore, orthogonalize, orthogonalize_monthend,
    AShareConsensus, FamaFrench3Factor
)
from quant.core import MinuteBarLoader
from quant.factor_test import FactorTester
from quant.risk_model import RiskModelManager
from quant.portfolio_test import PortfolioAnalyzer
from quant.feature import MinuteBar, MinuteBarFeatureManager


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
    dm.update_stock_quote(20050101)
    dm.update_stock_size(20050101)
    dm.update_index_quote(20050101)
    dm.update_risk_model()

    # DataProcessor
    dp = DataProcessor([AShareConsensus, FamaFrench3Factor])
    dp.run()
    
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
    factor = pd.read_pickle('D:/peiyq/民生/权重巨振/data/factor/processed/amount_beta/amount_beta_mean20.pkl')
    factor = factor.pivot(index='trade_date', columns='ticker', values='factor')
    factor = format_unstack_table(factor)

    universe = univ(listed_days=120, continuous_trade_days=20, include_st=False, include_suspend=False, include_price_limit=False, index_code='399303.SZ')
    ft = FactorTester(universe, 'M', 20161230, 20231229, 'vwap')

    factor_stan = stdd_zscore(winsorize_mad(factor))

    factor_orth = orthogonalize(factor_stan, ft.risk_model['lncap'], *list(map(lambda x: ft._prepare_industry(x), ft.RISK_INDUSTRY_FACTORS)))
    # factor_orth = orthogonalize_monthend(factor_stan, ft.risk_model['lncap'], *list(map(lambda x: ft._prepare_industry(x), ft.RISK_INDUSTRY_FACTORS)))

    output = ft.test(-factor_orth, 20, True)

    # Risk Model
    rmm = RiskModelManager('./risk_model/structure.yaml', init_date=20231101)
    factor = rmm.calc_risk_subfactor(rmm.structure[10]['subfactors'][0])
<<<<<<< HEAD
    factor = rmm.calc_risk_factor(rmm.structure[9])
    factor = rmm.calc_exposure()

    # Portfolio Analyzer
    pa = PortfolioAnalyzer()
    
    # minute bar
    bar_loader = MinuteBarLoader('1m')
    data = bar_loader[20230101:20231231]
    bar = MinuteBar(data[100])

    # minute bar calculation
    mbf = MinuteBarFeatureManager()
=======
    # factor = rmm.calc_exposure()

    # Portfolio Analyzer
    from quant.portfolio_test import PortfolioAnalyzer
    pa = PortfolioAnalyzer()
>>>>>>> 32869e41ae704f69ea6cde5ecd138652d7d4b8ce
