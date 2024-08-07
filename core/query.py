# -*- coding: utf-8 -*-
"""
# @Time : 2022/5/01 15:53
# @Author: Yao Yifan
# @File : query.py

数据库查询：
需要继承自WindQuery或DatayesQuery:
1.如果为全量数据同步，则仅需要写明sql
2.如果为增量维护类数据，需要指定任务类型task = 'incremental',并且指定日期字段field_date，可以为DataFrame指定索引字段field_date。
按日期查询的条件语句由父类中的query方法自动添加，无需编写进sql中

类名称会作为本地pickle文件的名称，避免重名
"""


# %% Wind Query
class WindQuery:
    """ 0
    Wind数据库查询
    """
    task = 'full'
    sql = None
    field_omit = ['OBJECT_ID']
    field_index = None
    field_date = None

    @classmethod
    def query(cls, date=None):
        assert cls.task in ('full', 'incremental')
        if cls.task == 'full':
            return cls.sql
        else:
            assert cls.field_date is not None
            if 'where' in cls.sql:
                return f"{cls.sql} and {cls.field_date} > '{date}'"
            else:
                return f"{cls.sql} where {cls.field_date} > '{date}'"


class AShareCalendar(WindQuery):
    sql = '''
        select TRADE_DAYS from winddb.AShareCalendar 
        where TRADE_DAYS<=(select max(TRADE_DT) from winddb.AShareEODPrices) 
        AND TRADE_DAYS > 20041231 AND S_INFO_EXCHMARKET =  'SSE' 
        order by TRADE_DAYS
        '''
    
class AShareCalendarAll(WindQuery):
    sql = '''
        select TRADE_DAYS from winddb.AShareCalendar 
        where TRADE_DAYS > 20041231 AND S_INFO_EXCHMARKET =  'SSE' 
        order by TRADE_DAYS
        '''

class AShareDescription(WindQuery):
    sql = '''
        select S_INFO_WINDCODE, S_INFO_NAME, S_INFO_LISTDATE, S_INFO_DELISTDATE 
        from winddb.AShareDescription
        where S_INFO_LISTDATE is not null and S_INFO_WINDCODE <> 'T00018.SH'
        '''

class AShareEODPrices(WindQuery):
    task = 'incremental'
    field_index = ['TRADE_DT', 'S_INFO_WINDCODE']
    field_date = 'TRADE_DT'
    sql = '''
        select TRADE_DT, S_INFO_WINDCODE, S_DQ_PRECLOSE, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, 
                S_DQ_CLOSE, S_DQ_PCTCHANGE, S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_AVGPRICE,
                S_DQ_ADJFACTOR, S_DQ_TRADESTATUSCODE, S_DQ_LIMIT, S_DQ_STOPPING
        from winddb.AShareEODPrices
        '''

class AShareEODDerivativeIndicator(AShareEODPrices):
    sql = '''
           select TRADE_DT, S_INFO_WINDCODE, S_VAL_MV, S_DQ_MV, TOT_SHR_TODAY, FLOAT_A_SHR_TODAY, FREE_SHARES_TODAY, S_VAL_PB_NEW, S_VAL_PE_TTM
           from winddb.AShareEODDerivativeIndicator
           '''

class AShareMoneyFlow(AShareEODPrices):
    sql = '''
            select TRADE_DT, S_INFO_WINDCODE, BUY_VALUE_EXLARGE_ORDER, SELL_VALUE_EXLARGE_ORDER,
            BUY_VALUE_LARGE_ORDER, SELL_VALUE_LARGE_ORDER, BUY_VALUE_MED_ORDER, SELL_VALUE_MED_ORDER,
            BUY_VALUE_SMALL_ORDER, SELL_VALUE_SMALL_ORDER, OPEN_NET_INFLOW_RATE_VALUE_L
            from winddb.AShareMoneyFlow
            '''

class AShareBalanceSheet(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DT'
    sql = '''
        select * from winddb.AShareBalanceSheet 
        where STATEMENT_TYPE in 
            ('408001000','408004000','408005000','408019000','408027000','408028000','408029000','408050000')
    '''

class AShareCashFlow(AShareBalanceSheet):
    sql = '''
           select * from winddb.AShareCashFlow 
           where STATEMENT_TYPE in 
               ('408001000','408004000','408005000','408019000','408027000','408028000','408029000','408050000')
       '''

class AShareIncome(AShareBalanceSheet):
    sql = '''
           select * from winddb.AShareIncome 
           where STATEMENT_TYPE in 
               ('408001000','408004000','408005000','408019000','408027000','408028000','408029000','408050000')
       '''

class AShareProfitExpress(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DT'
    sql = '''
           select * from winddb.AShareProfitExpress            
            '''

class AShareProfitNotice(WindQuery):
    task = 'incremental'
    field_date = 'S_PROFITNOTICE_DATE'
    sql = '''
           select * from winddb.AShareProfitNotice            
            '''
            
class AShareANNFinancialIndicator(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DT'
    sql = '''select * from winddb.AShareANNFinancialIndicator'''

class AShareFinancialIndicator(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DT'
    sql = '''
           select * from winddb.AShareFinancialIndicator            
            '''

class AShareTTMAndMRQ(WindQuery):
    task = 'incremental'
    field_date = 'REPORT_PERIOD'
    sql = '''
           select * from winddb.AShareTTMAndMRQ            
            '''
    

class AShareFinancialDerivative(WindQuery):
    sql = '''select * from winddb.AShareFinancialDerivative'''

class AShareDividend(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DT'
    sql = '''
           select * from winddb.AShareDividend            
            '''

class AShareEarningEst(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DT'
    sql = '''
            select S_INFO_WINDCODE, EST_DT, ANN_DT, REPORTING_PERIOD, RESEARCH_INST_NAME, ANALYST_NAME, 
                EST_NET_PROFIT, EST_MAIN_BUS_INC, S_EST_ROA, S_EST_ROE
            from winddb.AShareEarningEst            
            '''

class AShareConsensusData(WindQuery):
    task = 'incremental'
    field_date = 'EST_DT'
    sql = '''
            select  EST_DT, S_INFO_WINDCODE, EST_REPORT_DT, CONSEN_DATA_CYCLE_TYP, 
                    NUM_EST_INST, EPS_AVG, NET_PROFIT_AVG, MAIN_BUS_INC_AVG
            from winddb.AShareConsensusData            
            '''

class AShareConsensusRollingData(WindQuery):
    task = 'incremental'
    field_date = 'EST_DT'
    field_index = ['EST_DT', 'S_INFO_WINDCODE', 'ROLLING_TYPE']
    sql = '''
            select  EST_DT, S_INFO_WINDCODE, ROLLING_TYPE, NET_PROFIT, EST_EPS， EST_ROE，EST_OPER_REVENUE
            from winddb.AShareConsensusRollingData    
            where ROLLING_TYPE in ('FTTM','FY0','FY1','FY2','FY3')        
            '''

class AShareHolderNumber(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DT'
    sql = """
        select S_INFO_WINDCODE, ANN_DT, S_HOLDER_ENDDATE, S_HOLDER_NUM, S_HOLDER_TOTAL_NUM
        from winddb.AShareHolderNumber
        """

class AShareIndustriesClassCITICS(WindQuery):
    sql = '''
    select a.S_INFO_WINDCODE, b.INDUSTRIESNAME, a.ENTRY_DT, a.REMOVE_DT, a.CUR_SIGN from winddb.AShareIndustriesClassCITICS a, winddb.AShareIndustriesCode b
    where substr(a.CITICS_IND_CODE, 1, 4) = substr(b.INDUSTRIESCODE, 1, 4) and b.LEVELNUM = '2'
    order by S_INFO_WINDCODE 
    '''

class AShareIndustriesClassCITICS3(WindQuery):
    sql = '''
    select a.S_INFO_WINDCODE, b.INDUSTRIESNAME, a.ENTRY_DT, a.REMOVE_DT, a.CUR_SIGN from winddb.AShareIndustriesClassCITICS a, winddb.AShareIndustriesCode b
    where substr(a.CITICS_IND_CODE, 1, 8) = substr(b.INDUSTRIESCODE, 1, 8) and b.LEVELNUM = '4'
    order by S_INFO_WINDCODE 
    '''

class AShareEquityRelationships(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DT'
    sql = '''
    select S_INFO_WINDCODE, ANN_DT, ACTUALCONTROLLER_TYPE from winddb.AShareEquityRelationships
    '''

class AShareInsideHolder(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DT'
    sql = '''
    select S_INFO_WINDCODE, ANN_DT, S_HOLDER_ENDDATE, S_HOLDER_NAME, S_HOLDER_QUANTITY, S_HOLDER_PCT, S_HOLDER_SHARECATEGORY,
    S_HOLDER_RESTRICTEDQUANTITY, S_HOLDER_ANAME, S_HOLDER_SHARECATEGORYNAME, REPORT_PERIOD from winddb.AShareInsideHolder'''

class AShareCapitalization(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DT'
    sql = '''
    select * from winddb.AShareCapitalization'''

class AShareST(WindQuery):
    sql = '''select S_INFO_WINDCODE, S_TYPE_ST, ENTRY_DT, REMOVE_DT, ANN_DT, OPDATE from winddb.AShareST'''

class AShareTTMHis(WindQuery):
    sql = '''select * from winddb.AShareTTMHis'''

class AShareSWNIndustriesClass(WindQuery):
    sql = '''
    select a.S_INFO_WINDCODE, b.INDUSTRIESNAME, a.ENTRY_DT, a.REMOVE_DT from winddb.AShareSWNIndustriesClass a, AShareIndustriesCode b
    where substr(a.SW_IND_CODE, 1, 4) = substr(b.INDUSTRIESCODE, 1, 4) and b.LEVELNUM = '2'
    order by S_INFO_WINDCODE 
    '''

class AShareMarginSubject(WindQuery):
    sql = """SELECT * FROM winddb.AShareMarginSubject"""
    
class AIndexEODPrices(WindQuery):
    task = 'incremental'
    field_index = ['TRADE_DT', 'S_INFO_WINDCODE']
    field_date = 'TRADE_DT'
    field_code = ('000001.SH', '000002.SH', '000300.SH', '000905.SH', '000985.CSI', '000906.SH', '000852.SH',
                  '399006.SZ', '399303.SZ', '000688.SH', '000016.SH', '932000.CSI', '399102.SZ', '000919.CSI',
                  '000918.CSI', '000922.CSI', 'h30356.CSI', 'h30335.CSI', '399371.SZ', '399370.SZ', '399372.SZ',
                  '399373.SZ', '399376.SZ', '399377.SZ', '399417.SZ', '930997.CSI', '399808.SZ', '931643.CSI',
                  '000698.SH', '931643.CSI', '399324.SZ', '399106.SZ', '399107.SZ', '399673.SZ', '399295.SZ',
                  '399296.SZ', '399001.SZ', '930050.CSI', 'h00922.CSI', '931446.CSI', '921446.CSI', '931130.CSI',
                  'h21130.CSI', '930955.CSI', 'h20955.CSI', 'h30269.CSI', 'h20269.CSI', '930740.CSI', 'h20740.CSI',
                  'h00300.CSI')
    sql = f'''
    select TRADE_DT, S_INFO_WINDCODE, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_PCTCHANGE, S_DQ_VOLUME, 
    S_DQ_AMOUNT from winddb.AIndexEODPrices 
    where S_INFO_WINDCODE IN {field_code}
    '''

class AIndexValuation(WindQuery):
    task = 'incremental'
    field_index = ['TRADE_DT', 'S_INFO_WINDCODE']
    field_date = 'TRADE_DT'
    field_code = ('000001.SH', '000002.SH', '000300.SH', '000905.SH', '000985.CSI', '000906.SH', '000852.SH',
                  '399006.SZ', '399303.SZ', '000688.SH', '000016.SH', '932000.CSI', '399102.SZ', '000919.CSI',
                  '000918.CSI', '000922.CSI', 'h30356.CSI', 'h30335.CSI', '399371.SZ', '399370.SZ', '399372.SZ',
                  '399373.SZ', '399376.SZ', '399377.SZ', '399417.SZ', '930997.CSI', '399808.SZ', '931643.CSI',
                  '000698.SH', '931643.CSI', '399324.SZ', '399106.SZ', '399107.SZ', '399673.SZ', '399295.SZ',
                  '399296.SZ', '399001.SZ', '930050.CSI', 'h00922.CSI', '931446.CSI', '921446.CSI', '931130.CSI',
                  'h21130.CSI', '930955.CSI', 'h20955.CSI', 'h30269.CSI', 'h20269.CSI', '930740.CSI', 'h20740.CSI',
                  'h00300.CSI')
    sql = f'''
    select TRADE_DT, S_INFO_WINDCODE, MV_TOTAL, MV_FLOAT, TOT_SHR, TOT_SHR_FLOAT, TOT_SHR_FREE, TURNOVER,
    TURNOVER_FREE, PE_TTM, PB_LF from winddb.AIndexValuation 
    where S_INFO_WINDCODE IN {field_code}
    '''

class AIndexValuationIndustriesCITICS3(WindQuery):
    task = 'incremental'
    field_index = ['TRADE_DT', 'S_INFO_WINDCODE']
    field_date = 'TRADE_DT'
    field_code = ('CI005201.WI','CI005202.WI','CI005205.WI','CI005206.WI','CI005207.WI','CI005396.WI','CI005397.WI','CI005208.WI','CI005209.WI','CI005210.WI','CI005398.WI','CI005211.WI','CI005212.WI','CI005213.WI','CI005214.WI','CI005215.WI','CI005216.WI','CI005217.WI','CI005218.WI','CI005219.WI','CI005399.WI','CI005400.WI','CI005401.WI','CI005402.WI','CI005403.WI','CI005220.WI','CI005221.WI','CI005222.WI','CI005223.WI','CI005404.WI','CI005405.WI','CI005224.WI','CI005225.WI','CI005226.WI','CI005227.WI','CI005406.WI','CI005228.WI','CI005229.WI','CI005407.WI','CI005230.WI','CI005231.WI','CI005232.WI','CI005231.WI','CI005408.WI','CI005409.WI','CI005230.WI','CI005233.WI','CI005234.WI','CI005235.WI','CI005236.WI','CI005237.WI','CI005410.WI','CI005411.WI','CI005412.WI','CI005238.WI','CI005239.WI','CI005240.WI','CI005241.WI','CI005242.WI','CI005243.WI','CI005244.WI','CI005245.WI','CI005246.WI','CI005247.WI','CI005248.WI','CI005249.WI','CI005250.WI','CI005251.WI','CI005203.WI','CI005204.WI','CI005252.WI','CI005253.WI','CI005254.WI','CI005255.WI','CI005256.WI','CI005238.WI','CI005240.WI','CI005242.WI','CI005413.WI','CI005414.WI','CI005415.WI','CI005416.WI','CI005417.WI','CI005418.WI','CI005419.WI','CI005247.WI','CI005248.WI','CI005250.WI','CI005252.WI','CI005253.WI','CI005255.WI','CI005420.WI','CI005421.WI','CI005422.WI','CI005423.WI','CI005424.WI','CI005425.WI','CI005426.WI','CI005427.WI','CI005428.WI','CI005429.WI','CI005430.WI','CI005431.WI','CI005257.WI','CI005432.WI','CI005433.WI','CI005434.WI','CI005435.WI','CI005258.WI','CI005436.WI','CI005437.WI','CI005259.WI','CI005260.WI','CI005261.WI','CI005262.WI','CI005259.WI','CI005438.WI','CI005439.WI','CI005261.WI','CI005440.WI','CI005441.WI','CI005442.WI','CI005263.WI','CI005264.WI','CI005265.WI','CI005266.WI','CI005443.WI','CI005444.WI','CI005445.WI','CI005446.WI','CI005447.WI','CI005448.WI','CI005449.WI','CI005450.WI','CI005267.WI','CI005451.WI','CI005452.WI','CI005453.WI','CI005268.WI','CI005269.WI','CI005270.WI','CI005271.WI','CI005272.WI','CI005273.WI','CI005274.WI','CI005275.WI','CI005276.WI','CI005277.WI','CI005278.WI','CI005470.WI','CI005279.WI','CI005280.WI','CI005454.WI','CI005455.WI','CI005456.WI','CI005457.WI','CI005458.WI','CI005459.WI','CI005460.WI','CI005461.WI','CI005271.WI','CI005462.WI','CI005463.WI','CI005464.WI','CI005465.WI','CI005466.WI','CI005467.WI','CI005468.WI','CI005469.WI','CI005471.WI','CI005281.WI','CI005282.WI','CI005283.WI','CI005284.WI','CI005285.WI','CI005286.WI','CI005472.WI','CI005473.WI','CI005474.WI','CI005475.WI','CI005284.WI','CI005286.WI','CI005476.WI','CI005477.WI','CI005478.WI','CI005479.WI','CI005480.WI','CI005481.WI','CI005482.WI','CI005287.WI','CI005288.WI','CI005289.WI','CI005290.WI','CI005291.WI','CI005292.WI','CI005293.WI','CI005294.WI','CI005295.WI','CI005296.WI','CI005297.WI','CI005298.WI','CI005299.WI','CI005300.WI','CI005301.WI','CI005299.WI','CI005483.WI','CI005484.WI','CI005485.WI','CI005486.WI','CI005487.WI','CI005488.WI','CI005489.WI','CI005490.WI','CI005491.WI','CI005302.WI','CI005303.WI','CI005492.WI','CI005493.WI','CI005494.WI','CI005304.WI','CI005305.WI','CI005495.WI','CI005496.WI','CI005497.WI','CI005498.WI','CI005499.WI','CI005501.WI','CI005502.WI','CI005306.WI','CI005307.WI','CI005308.WI','CI005309.WI','CI005310.WI','CI005503.WI','CI005309.WI','CI005504.WI','CI005505.WI','CI005386.WI','CI005387.WI','CI005388.WI','CI005389.WI','CI005506.WI','CI005390.WI','CI005391.WI','CI005393.WI','CI005394.WI','CI005393.WI','CI005394.WI','CI005507.WI','CI005508.WI','CI005509.WI','CI005510.WI','CI005511.WI','CI005316.WI','CI005317.WI','CI005318.WI','CI005319.WI','CI005320.WI','CI005321.WI','CI005322.WI','CI005323.WI','CI005324.WI','CI005325.WI','CI005512.WI','CI005325.WI','CI005326.WI','CI005327.WI','CI005328.WI','CI005329.WI','CI005330.WI','CI005331.WI','CI005332.WI','CI005328.WI','CI005331.WI','CI005330.WI','CI005513.WI','CI005514.WI','CI005515.WI','CI005516.WI','CI005333.WI','CI005334.WI','CI005335.WI','CI005336.WI','CI005337.WI','CI005338.WI','CI005518.WI','CI005519.WI','CI005339.WI','CI005340.WI','CI005341.WI','CI005521.WI','CI005333.WI','CI005517.WI','CI005520.WI','CI005522.WI','CI005523.WI','CI005342.WI','CI005343.WI','CI005344.WI','CI005344.WI','CI005524.WI','CI005345.WI','CI005346.WI','CI005347.WI','CI005348.WI','CI005347.WI','CI005525.WI','CI005526.WI','CI005349.WI','CI005350.WI','CI005351.WI','CI005352.WI','CI005528.WI','CI005529.WI','CI005353.WI','CI005354.WI','CI005355.WI','CI005355.WI','CI005356.WI','CI005536.WI','CI005537.WI','CI005357.WI','CI005358.WI','CI005359.WI','CI005360.WI','CI005361.WI','CI005362.WI','CI005363.WI','CI005538.WI','CI005539.WI','CI005540.WI','CI005541.WI','CI005542.WI','CI005543.WI','CI005544.WI','CI005545.WI','CI005546.WI','CI005547.WI','CI005548.WI','CI005549.WI','CI005550.WI','CI005364.WI','CI005375.WI','CI005376.WI','CI005377.WI','CI005378.WI','CI005379.WI','CI005380.WI','CI005381.WI','CI005553.WI','CI005554.WI','CI005555.WI','CI005556.WI','CI005366.WI','CI005551.WI','CI005557.WI','CI005558.WI','CI005559.WI','CI005367.WI','CI005368.WI','CI005382.WI','CI005383.WI','CI005384.WI','CI005385.WI','CI005560.WI','CI005561.WI','CI005562.WI','CI005563.WI','CI005564.WI','CI005565.WI','CI005566.WI','CI005567.WI','CI005568.WI','CI005569.WI','CI005570.WI','CI005571.WI','CI005369.WI','CI005370.WI','CI005371.WI','CI005372.WI','CI005373.WI','CI005369.WI','CI005572.WI','CI005573.WI','CI005574.WI','CI005575.WI','CI005576.WI','CI005577.WI','CI005578.WI','CI005579.WI','CI005580.WI','CI005581.WI','CI005374.WI','CI005395.WI','CI005530.WI','CI005531.WI','CI005533.WI')
    sql = f'''
    select TRADE_DT, S_INFO_WINDCODE, MV_TOTAL, MV_FLOAT, TOT_SHR, TOT_SHR_FLOAT, TOT_SHR_FREE, TURNOVER,
    TURNOVER_FREE, PE_TTM, PB_LF from winddb.AIndexValuation 
    where S_INFO_WINDCODE IN {field_code}
    '''
    
class AIndexMembers(WindQuery):
    field_code = ('000001.SH', '000002.SH', '000300.SH', '000905.SH', '000985.CSI', '000906.SH', '000852.SH',
                  '399006.SZ', '399303.SZ', '000688.SH', '000016.SH', '932000.CSI', '399102.SZ', '000919.CSI',
                  '000918.CSI', '000922.CSI', 'h30356.CSI', 'h30335.CSI', '399371.SZ', '399370.SZ', '399372.SZ',
                  '399373.SZ', '399376.SZ', '399377.SZ', '399417.SZ', '930997.CSI', '399808.SZ', '931643.CSI',
                  '000698.SH', '931643.CSI', '399324.SZ', '399106.SZ', '399107.SZ', '399673.SZ', '399295.SZ',
                  '399296.SZ', '399001.SZ', '930050.CSI', 'h00922.CSI', '931446.CSI', '921446.CSI', '931130.CSI',
                  'h21130.CSI', '930955.CSI', 'h20955.CSI', 'h30269.CSI', 'h20269.CSI', '930740.CSI', 'h20740.CSI',
                  'h00300.CSI')
    sql = f'''select * from winddb.AIndexMembers where S_INFO_WINDCODE in {field_code}'''


class AIndexHS300FreeWeight(WindQuery):
    task = 'incremental'
    field_index = ['TRADE_DT', 'S_INFO_WINDCODE', 'S_CON_WINDCODE']
    field_date = 'TRADE_DT'
    field_code = ('000001.SH', '000002.SH', '000300.SH', '000905.SH', '000985.CSI', '000906.SH', '000852.SH',
                  '399006.SZ', '399303.SZ', '000688.SH', '000016.SH', '932000.CSI', '399102.SZ', '000919.CSI',
                  '000918.CSI', '000922.CSI', 'h30356.CSI', 'h30335.CSI', '399371.SZ', '399370.SZ', '399372.SZ',
                  '399373.SZ', '399376.SZ', '399377.SZ', '399417.SZ', '930997.CSI', '399808.SZ', '931643.CSI',
                  '000698.SH', '931643.CSI', '399324.SZ', '399106.SZ', '399107.SZ', '399673.SZ', '399295.SZ',
                  '399296.SZ', '399001.SZ', '930050.CSI', 'h00922.CSI', '931446.CSI', '921446.CSI', '931130.CSI',
                  'h21130.CSI', '930955.CSI', 'h20955.CSI', 'h30269.CSI', 'h20269.CSI', '930740.CSI', 'h20740.CSI',
                  'h00300.CSI')
    sql = f'''select S_INFO_WINDCODE, S_CON_WINDCODE, TRADE_DT, I_WEIGHT from winddb.AIndexHS300FreeWeight where S_INFO_WINDCODE in {field_code}'''

class AIndexHS300CloseWeight(WindQuery):
    task = 'incremental'
    field_date = 'TRADE_DT'
    sql = '''
    select TRADE_DT, S_INFO_WINDCODE, S_CON_WINDCODE, I_WEIGHT from winddb.AIndexHS300CloseWeight
    '''

class AIndexIndustriesEODCitics(WindQuery):
    task = 'incremental'
    field_index = ['TRADE_DT', 'S_INFO_WINDCODE']
    field_date = 'TRADE_DT'
    sql = '''
        select TRADE_DT, S_INFO_WINDCODE, S_DQ_PRECLOSE, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, 
                S_DQ_CLOSE, S_DQ_PCTCHANGE, S_DQ_VOLUME, S_DQ_AMOUNT
        from winddb.AIndexIndustriesEODCitics
        '''

class AIndexConsensusRollingData(WindQuery):
    task = 'incremental'
    field_date = 'EST_DT'
    field_index = ['EST_DT', 'S_INFO_WINDCODE', 'ROLLING_TYPE']
    # 中信一级行业
    # field_code = （'CI005006.WI', 'CI005017.WI', 'CI005024.WI', 'CI005002.WI',
    # 'CI005018.WI', 'CI005005.WI', 'CI005003.WI', 'CI005009.WI',
    # 'CI005004.WI', 'CI005010.WI', 'CI005014.WI', 'CI005025.WI',
    # 'CI005013.WI', 'CI005023.WI', 'CI005016.WI', 'CI005015.WI',
    # 'CI005027.WI', 'CI005007.WI', 'CI005020.WI', 'CI005011.WI',
    # 'CI005008.WI', 'CI005019.WI', 'CI005001.WI', 'CI005021.WI',
    # 'CI005026.WI', 'CI005022.WI', 'CI005029.WI', 'CI005028.WI'）
    # 中信三级行业
    field_code = ('CI005201.WI','CI005202.WI','CI005205.WI','CI005206.WI','CI005207.WI','CI005396.WI','CI005397.WI','CI005208.WI','CI005209.WI','CI005210.WI','CI005398.WI','CI005211.WI','CI005212.WI','CI005213.WI','CI005214.WI','CI005215.WI','CI005216.WI','CI005217.WI','CI005218.WI','CI005219.WI','CI005399.WI','CI005400.WI','CI005401.WI','CI005402.WI','CI005403.WI','CI005220.WI','CI005221.WI','CI005222.WI','CI005223.WI','CI005404.WI','CI005405.WI','CI005224.WI','CI005225.WI','CI005226.WI','CI005227.WI','CI005406.WI','CI005228.WI','CI005229.WI','CI005407.WI','CI005230.WI','CI005231.WI','CI005232.WI','CI005231.WI','CI005408.WI','CI005409.WI','CI005230.WI','CI005233.WI','CI005234.WI','CI005235.WI','CI005236.WI','CI005237.WI','CI005410.WI','CI005411.WI','CI005412.WI','CI005238.WI','CI005239.WI','CI005240.WI','CI005241.WI','CI005242.WI','CI005243.WI','CI005244.WI','CI005245.WI','CI005246.WI','CI005247.WI','CI005248.WI','CI005249.WI','CI005250.WI','CI005251.WI','CI005203.WI','CI005204.WI','CI005252.WI','CI005253.WI','CI005254.WI','CI005255.WI','CI005256.WI','CI005238.WI','CI005240.WI','CI005242.WI','CI005413.WI','CI005414.WI','CI005415.WI','CI005416.WI','CI005417.WI','CI005418.WI','CI005419.WI','CI005247.WI','CI005248.WI','CI005250.WI','CI005252.WI','CI005253.WI','CI005255.WI','CI005420.WI','CI005421.WI','CI005422.WI','CI005423.WI','CI005424.WI','CI005425.WI','CI005426.WI','CI005427.WI','CI005428.WI','CI005429.WI','CI005430.WI','CI005431.WI','CI005257.WI','CI005432.WI','CI005433.WI','CI005434.WI','CI005435.WI','CI005258.WI','CI005436.WI','CI005437.WI','CI005259.WI','CI005260.WI','CI005261.WI','CI005262.WI','CI005259.WI','CI005438.WI','CI005439.WI','CI005261.WI','CI005440.WI','CI005441.WI','CI005442.WI','CI005263.WI','CI005264.WI','CI005265.WI','CI005266.WI','CI005443.WI','CI005444.WI','CI005445.WI','CI005446.WI','CI005447.WI','CI005448.WI','CI005449.WI','CI005450.WI','CI005267.WI','CI005451.WI','CI005452.WI','CI005453.WI','CI005268.WI','CI005269.WI','CI005270.WI','CI005271.WI','CI005272.WI','CI005273.WI','CI005274.WI','CI005275.WI','CI005276.WI','CI005277.WI','CI005278.WI','CI005470.WI','CI005279.WI','CI005280.WI','CI005454.WI','CI005455.WI','CI005456.WI','CI005457.WI','CI005458.WI','CI005459.WI','CI005460.WI','CI005461.WI','CI005271.WI','CI005462.WI','CI005463.WI','CI005464.WI','CI005465.WI','CI005466.WI','CI005467.WI','CI005468.WI','CI005469.WI','CI005471.WI','CI005281.WI','CI005282.WI','CI005283.WI','CI005284.WI','CI005285.WI','CI005286.WI','CI005472.WI','CI005473.WI','CI005474.WI','CI005475.WI','CI005284.WI','CI005286.WI','CI005476.WI','CI005477.WI','CI005478.WI','CI005479.WI','CI005480.WI','CI005481.WI','CI005482.WI','CI005287.WI','CI005288.WI','CI005289.WI','CI005290.WI','CI005291.WI','CI005292.WI','CI005293.WI','CI005294.WI','CI005295.WI','CI005296.WI','CI005297.WI','CI005298.WI','CI005299.WI','CI005300.WI','CI005301.WI','CI005299.WI','CI005483.WI','CI005484.WI','CI005485.WI','CI005486.WI','CI005487.WI','CI005488.WI','CI005489.WI','CI005490.WI','CI005491.WI','CI005302.WI','CI005303.WI','CI005492.WI','CI005493.WI','CI005494.WI','CI005304.WI','CI005305.WI','CI005495.WI','CI005496.WI','CI005497.WI','CI005498.WI','CI005499.WI','CI005501.WI','CI005502.WI','CI005306.WI','CI005307.WI','CI005308.WI','CI005309.WI','CI005310.WI','CI005503.WI','CI005309.WI','CI005504.WI','CI005505.WI','CI005386.WI','CI005387.WI','CI005388.WI','CI005389.WI','CI005506.WI','CI005390.WI','CI005391.WI','CI005393.WI','CI005394.WI','CI005393.WI','CI005394.WI','CI005507.WI','CI005508.WI','CI005509.WI','CI005510.WI','CI005511.WI','CI005316.WI','CI005317.WI','CI005318.WI','CI005319.WI','CI005320.WI','CI005321.WI','CI005322.WI','CI005323.WI','CI005324.WI','CI005325.WI','CI005512.WI','CI005325.WI','CI005326.WI','CI005327.WI','CI005328.WI','CI005329.WI','CI005330.WI','CI005331.WI','CI005332.WI','CI005328.WI','CI005331.WI','CI005330.WI','CI005513.WI','CI005514.WI','CI005515.WI','CI005516.WI','CI005333.WI','CI005334.WI','CI005335.WI','CI005336.WI','CI005337.WI','CI005338.WI','CI005518.WI','CI005519.WI','CI005339.WI','CI005340.WI','CI005341.WI','CI005521.WI','CI005333.WI','CI005517.WI','CI005520.WI','CI005522.WI','CI005523.WI','CI005342.WI','CI005343.WI','CI005344.WI','CI005344.WI','CI005524.WI','CI005345.WI','CI005346.WI','CI005347.WI','CI005348.WI','CI005347.WI','CI005525.WI','CI005526.WI','CI005349.WI','CI005350.WI','CI005351.WI','CI005352.WI','CI005528.WI','CI005529.WI','CI005353.WI','CI005354.WI','CI005355.WI','CI005355.WI','CI005356.WI','CI005536.WI','CI005537.WI','CI005357.WI','CI005358.WI','CI005359.WI','CI005360.WI','CI005361.WI','CI005362.WI','CI005363.WI','CI005538.WI','CI005539.WI','CI005540.WI','CI005541.WI','CI005542.WI','CI005543.WI','CI005544.WI','CI005545.WI','CI005546.WI','CI005547.WI','CI005548.WI','CI005549.WI','CI005550.WI','CI005364.WI','CI005375.WI','CI005376.WI','CI005377.WI','CI005378.WI','CI005379.WI','CI005380.WI','CI005381.WI','CI005553.WI','CI005554.WI','CI005555.WI','CI005556.WI','CI005366.WI','CI005551.WI','CI005557.WI','CI005558.WI','CI005559.WI','CI005367.WI','CI005368.WI','CI005382.WI','CI005383.WI','CI005384.WI','CI005385.WI','CI005560.WI','CI005561.WI','CI005562.WI','CI005563.WI','CI005564.WI','CI005565.WI','CI005566.WI','CI005567.WI','CI005568.WI','CI005569.WI','CI005570.WI','CI005571.WI','CI005369.WI','CI005370.WI','CI005371.WI','CI005372.WI','CI005373.WI','CI005369.WI','CI005572.WI','CI005573.WI','CI005574.WI','CI005575.WI','CI005576.WI','CI005577.WI','CI005578.WI','CI005579.WI','CI005580.WI','CI005581.WI','CI005374.WI','CI005395.WI','CI005530.WI','CI005531.WI','CI005533.WI')
    # 中证红利指数
    # field_code = ('000922.CSI')
    sql = f'''select * from winddb.AIndexConsensusRollingData where S_INFO_WINDCODE in {field_code} '''

class AIndexConsensusData(WindQuery):
    task = 'incremental'
    field_data = 'EST_DT'
    field_index = ['EST_DT', 'S_INFO_WINDCODE']
    sql = '''select * from winddb.AIndexConsensusData'''

class ASWSINDEXEOD(WindQuery):
    task = 'incremental'
    field_index = ['TRADE_DT', 'S_INFO_WINDCODE']
    field_date = 'TRADE_DT'
    sql = '''
        select TRADE_DT, S_INFO_WINDCODE, S_DQ_PRECLOSE, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, 
                S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_AMOUNT
        from winddb.ASWSINDEXEOD
        '''

class AWindIndexEODPrices(WindQuery):
    task = 'incremental'
    field_index = ['TRADE_DT', 'S_INFO_WINDCODE']
    field_date = 'TRADE_DT'
    field_code = ('881001.WI', '8841431.WI', '885001.WI', '885000.WI', '881034.WI')
    sql = f'''
    select TRADE_DT, S_INFO_WINDCODE, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_PCTCHANGE, S_DQ_VOLUME, 
    S_DQ_AMOUNT from winddb.AIndexWindIndustriesEOD 
    where S_INFO_WINDCODE IN {field_code}
    '''

class ChinaMutualFundDescription(WindQuery):
    task = 'full'
    sql = """
    SELECT F_INFO_WINDCODE,	F_INFO_NAME, F_INFO_CORP_FUNDMANAGEMENTCOMP, F_INFO_ISINITIAL,
    F_INFO_ISSUEDATE, F_INFO_DELISTDATE, F_INFO_SETUPDATE, F_INFO_BENCHMARK
    FROM winddb.ChinaMutualFundDescription 
    """

class ChinaMutualFundSector(WindQuery):
    task = 'full'
    sql = """
            SELECT
                F_INFO_WINDCODE, S_INFO_SECTOR, S_INFO_SECTORENTRYDT, S_INFO_SECTOREXITDT
            FROM
                winddb.ChinaMutualFundSector 
            WHERE
                S_INFO_SECTOR IN ( '2001010101000000', '2001010201000000', '2001010202000000', '2001010204000000' )
    """

class ChinaMutualFundReportDate(WindQuery):
    task = 'full'
    sql = """
            SELECT
                S_INFO_WINDCODE,
                START_DT,
                END_DT,
                S_STM_ACTUAL_ISSUINGDATE,
                S_STM_CORRECT_NUM,
                S_STM_CORRECT_ISSUINGDATE 
            FROM
                winddb.CMFIssuingDatePredict 
            WHERE
                S_INFO_WINDCODE IN ( SELECT F_INFO_WINDCODE FROM winddb.ChinaMutualFundDescription WHERE F_INFO_ISINITIAL = 1 ) 
                AND S_INFO_WINDCODE IN ( SELECT F_INFO_WINDCODE FROM winddb.ChinaMutualFundSector WHERE S_INFO_SECTOR IN ( '2001010101000000', '2001010201000000', '2001010202000000', '2001010204000000' ) )
                
        """

class ChinaMutualFundStockPortfolio(WindQuery):
    task = 'incremental'
    field_date = 'ANN_DATE'
    sql = """
           SELECT DISTINCT
                S_INFO_WINDCODE,F_PRT_ENDDATE,S_INFO_STOCKWINDCODE,F_PRT_STKVALUE,F_PRT_STKQUANTITY,
                F_PRT_STKVALUETONAV,ANN_DATE,STOCK_PER,FLOAT_SHR_PER
           FROM            	
            winddb.ChinaMutualFundStockPortfolio S            	
           WHERE
            S.S_INFO_WINDCODE in (Select F_INFO_WINDCODE from winddb.ChinaMutualFundDescription where F_INFO_ISINITIAL = 1)
            AND S.S_INFO_WINDCODE in (Select F_INFO_WINDCODE from winddb.ChinaMutualFundSector where S_INFO_SECTOR in  ( '2001010101000000', '2001010201000000', '2001010202000000', '2001010204000000' ))
            AND S.ANN_DATE > S.F_PRT_ENDDATE             	           
            AND S.F_PRT_ENDDATE > 20041231                 
           """

class HkStockIndexMembers(WindQuery):
    sql = """SELECT * FROM winddb.HKStockIndexMembers"""
    
class HKShareEODPrices(AShareEODPrices):
    sql = '''
        select TRADE_DT, S_INFO_WINDCODE, S_DQ_PRECLOSE, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, 
                S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_AVGPRICE,
                S_DQ_ADJFACTOR
        from winddb.HKShareEODPrices
        '''
    
class HKShareEODDerivativeIndex(WindQuery):
    task = 'incremental'
    field_index = ['FINANCIAL_TRADE_DT', 'S_INFO_WINDCODE']
    field_date = 'FINANCIAL_TRADE_DT'
    sql = """SELECT * FROM winddb.HKShareEODDerivativeIndex"""
    
class SHSZRelatedSecurities(WindQuery):
    sql = """SELECT * FROM winddb.SHSZRelatedSecurities"""
