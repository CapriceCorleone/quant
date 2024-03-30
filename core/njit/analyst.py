'''
Author: WangXiang
Date: 2024-03-24 17:30:20
LastEditTime: 2024-03-29 21:40:36
'''

import numpy as np
from numba import njit


@njit
def replace_with_calendar(to_replace, calendar):
    diff_dates = set(to_replace).difference(set(calendar))
    replaced = to_replace.copy()
    for d in diff_dates:
        replaced[replaced == d] = calendar[calendar <= d][-1]
    return replaced


@njit
def afunc_consensus(m, annual_report, calendar, init_date, window=122, half_life=21):
    calendar = calendar.astype(np.float64)
    m = m[(~np.isnan(m[:, -1])) & (m[:, -1] != 0)]
    m[:, 0] = replace_with_calendar(m[:, 0], calendar)

    lamb = np.log(0.5) / half_life
    trade_dates = calendar[calendar >= init_date]
    trade_dates = trade_dates[trade_dates >= max(m[:, 0].min(), annual_report[:, 0].min())]

    init_ix = np.where(calendar == trade_dates[0])[0][0]
    date_ix_dict = {calendar[i]: i for i in range(len(calendar))}

    consensus = np.zeros((len(trade_dates), 7), dtype=np.float64) * np.nan
    m_within_record = np.zeros((1, 1))
    t_last_period_record = 0
    stat_by_inst = np.zeros((1, 1))
    for i in range(len(trade_dates)):
        current_ix = date_ix_dict[trade_dates[i]]
        t_end = calendar[current_ix]
        t_start = calendar[max(init_ix + i - window, 0)]
        m_within = m[(m[:, 0] <= t_end) & (m[:, 0] >= t_start)]
        t_last_period = annual_report[annual_report[:, 0] <= t_end][-1, 1]
        t_last_annual_profit = annual_report[annual_report[:, 0] <= t_end][-1, 2]
        # Fetch consensus data within window
        if len(m_within) > 0:
            consensus[i, 0] = t_end
            consensus[i, 1] = t_last_period
            consensus[i, 2] = t_last_annual_profit
            if np.array_equal(m_within_record, m_within) and t_last_period_record == t_last_period:
                consensus[i, 1:-1] = consensus[i - 1, 1:-1]
            else:
                for k in [1, 2, 3]:
                    target_period = t_last_period + 10000 * k
                    m_within_target = m_within[m_within[:, 2] == target_period]
                    inst_list = np.unique(m_within_target[:, 1])
                    stat_by_inst = np.zeros((len(inst_list), 4))
                    for j in range(len(inst_list)):
                        m_current_inst = m_within_target[m_within_target[:, 1] == inst_list[j]]
                        if len(m_current_inst) > 0:
                            j_samples = m_current_inst[m_current_inst[:, 0].argsort()]
                            j_forecast = j_samples[-1, -1]
                            j_date = j_samples[-1, 0]
                            j_days_to_now = current_ix - date_ix_dict[j_date]
                            stat_by_inst[j] = np.array([inst_list[j], j_date, j_forecast, j_days_to_now])
                    stat_by_inst = stat_by_inst[stat_by_inst[:, 1] != 0]
                    if len(stat_by_inst) > 0:
                        weights = np.exp(lamb * stat_by_inst[:, -1])
                        weights = weights / np.sum(weights)
                        consensus[i, k + 2] = np.sum(stat_by_inst[:, 2] * weights)
                m_within_record = m_within
                t_last_period_record = t_last_period
            report_year_gap = int(t_end // 10000 - t_last_period // 10000)
            fy1 = consensus[i, report_year_gap + 2]
            fy2 = consensus[i, report_year_gap + 3]
            day_num = date_ix_dict[t_end] - date_ix_dict[calendar[calendar <= t_end // 10000 * 10000].max()]
            w = np.array([252 - day_num, day_num])
            w[w < 0] = 0
            w = w / w.sum()
            consensus[i, -1] = fy1 * w[0] + fy2 * w[1]
    return consensus[(consensus[:, 0] > 0) & np.isfinite(consensus[:, 1])]


@njit
def afunc_coverage(m, calendar, inint_date, window=122):
    calendar = calendar.astype(np.float64)
    m[:, 0] = replace_with_calendar(m[:, 0], calendar)
    trade_dates = calendar[calendar >= inint_date]
    trade_dates = trade_dates[trade_dates >= m[:, 0].min()]
    init_ix = np.where(calendar == trade_dates[0])[0][0]
    date_ix_dict = {calendar[i]: i for i in range(len(calendar))}
    coverage = np.zeros((len(trade_dates), 2), dtype=np.float64) * np.nan
    for i in range(len(trade_dates)):
        current_ix = date_ix_dict[trade_dates[i]]
        t_end = calendar[current_ix]
        t_start = calendar[max(init_ix + i - window, 0)]
        m_within = m[(m[:, 0] >= t_start) & (m[:, 0] <= t_end)]
        count = len(np.unique(m_within[:, 1]))
        coverage[i, 0] = t_end
        coverage[i, 1] = count
    return coverage