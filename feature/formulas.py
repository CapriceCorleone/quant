'''
Author: WangXiang
Date: 2024-04-14 02:40:08
LastEditTime: 2024-04-14 02:52:16
'''

import numpy as np
import pandas as pd


# %%
__daily_features__ = [
    'daily_bar',
    'weekly_bar',
    'risk_factor'
]

__minute_features__ = [
    # TODO
]

__all__ = __daily_features__ + __minute_features__


# %% Daily Bar Factor
def __daily_bar(AShareEODPrices, init_date = 20050101, return_adjfactor = False):
    __valid_fields__ =  ['open', 'high', 'low', 'close', 'vwap', 'volume', 'amount']
    if return_adjfactor:
        __valid_fields__.append('adjfactor')
    field_mapping = dict(zip([field.split('_')[-1].lower() for field in AShareEODPrices.columns], AShareEODPrices.columns))
    field_mapping['vwap'] = field_mapping.pop('avgprice')
    df = AShareEODPrices.loc[str(init_date):].copy()
    for f in __valid_fields__:
        if f not in ('volume', 'amount', 'adjfactor'):
            df[field_mapping[f]] = df[field_mapping[f]] * df['S_DQ_ADJFACTOR']
        if f in ('volume', 'amount'):
            df[field_mapping[f]] = np.log(df[field_mapping[f]] + 1)
    outputs = df[[field_mapping[k] for k in __valid_fields__]].astype(np.float32).unstack()
    outputs = {field: outputs[field_mapping[field]] for field in __valid_fields__}
    return outputs


def daily_bar(AShareEODPrices, init_date = 20050101):
    return __daily_bar(AShareEODPrices, init_date, return_adjfactor = False)


def weekly_bar(AShareEODPrices, init_date = 20050101):
    __valid_fields__ = ['open', 'high',' low', 'close', 'vwap', 'volume', 'amount']
    outputs_daily = __daily_bar(AShareEODPrices, init_date = init_date - 10000, return_adjfactor = True)
    outputs = {}
    adj_factor = outputs_daily.pop('adjfactor').loc[str(init_date):]
    for field in __valid_fields__:
        if field == 'open':
            result = outputs_daily[field].shift(4)
        elif field == 'close':
            result = outputs_daily[field].copy()
        elif field == 'high':
            result = outputs_daily[field].rolling(5).max()
        elif field == 'low':
            result = outputs_daily[field].rolling(5).min()
        elif field in ('volume', 'amount'):
            result = (np.exp(outputs_daily[field]) - 1).rolling(5).sum()
        field_5d = field + '5d'
        outputs[field_5d] = result.loc[str(init_date):]
    vwap5d = np.where(outputs['amount5d'] > 0, outputs['amount5d'] / outputs['volume5d'] * adj_factor * 10, outputs['close5d'])
    outputs['vwap5d'] = pd.DataFrame(vwap5d, columns=outputs['close5d'].columns, index=outputs['close5d'].index)
    return outputs
