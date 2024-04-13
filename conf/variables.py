'''
Author: WangXiang
Date: 2024-04-09 20:16:20
LastEditTime: 2024-04-13 15:43:16
'''


ANNUALIZE_MULTIPLIER = {
    'D': 252,
    'W': 52,
    'M': 12,
    'Q': 4,
    'Y': 1,
    5:   52,
    10:  26,
    20:  12
}

RISK_STYLE_FACTORS = ['size', 'beta', 'trend', 'liquidity', 'volatility', 'value', 'growth', 'nls', 'certainty', 'soe']

INDUSTRY_NAMES = {
    'zx': [
        '交通运输', '传媒', '农林牧渔', '医药', '商贸零售', '国防军工', '基础化工', '家电',
        '建材', '建筑', '房地产', '有色金属', '机械', '汽车', '消费者服务', '煤炭',
        '电力及公用事业', '电力设备及新能源', '电子', '石油石化', '纺织服装', '综合',
        '综合金融', '计算机', '轻工制造', '通信', '钢铁', '银行', '非银行金融', '食品饮料'
    ]
}

RISK_INDUSTRY_TYPE = 'zx'
RISK_INDUSTRY_FACTORS = INDUSTRY_NAMES[RISK_INDUSTRY_TYPE]