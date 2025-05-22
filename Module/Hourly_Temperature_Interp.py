# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Hourly_Temperature_Interp.py
@Time: 2023/12/1 20:49
@Function: 
"""

import pandas as pd
import numpy as np

# interp ---
def hr_temp_interp(df_meteor_cmip_bc, dt, delta_T):

    # tasmax = df_meteor_cmip_bc.loc[dt.date().strftime('%Y-%m-%d'), 'tmax'].values[0]
    # tasmin = df_meteor_cmip_bc.loc[dt.date().strftime('%Y-%m-%d'), 'tmin'].values[0]

    tasmax = df_meteor_cmip_bc.loc[dt.strftime('%Y-%m-%d'), 'tmax']
    tasmin = df_meteor_cmip_bc.loc[dt.strftime('%Y-%m-%d'), 'tmin']

    # temperature sine ---
    if 0 < dt.month < 3:
        t_tmax, t_tmin = 14, 6
    elif dt.month == 3:
        t_tmax, t_tmin = 14, 5
    elif dt.month in [4, 8, 9]:
        t_tmax, t_tmin = 15, 5
    elif dt.month in [5, 7]:
        t_tmax, t_tmin = 15, 4
    elif dt.month == 6:
        t_tmax, t_tmin = 16, 4
    elif dt.month in [10, 11]:
        t_tmax, t_tmin = 14, 6
    else:
        t_tmax, t_tmin = 14, 7

    tas_day = np.arange(24, dtype=float)

    # '''
    delta_tas = (tasmax - tasmin) / (t_tmax - t_tmin)

    # tas_day[tas_day > t_tmax] = tasmax + (t_tmax - tas_day[tas_day > t_tmax]) * delta_tas
    # tas_day[(tas_day > t_tmin) & (tas_day < t_tmax)] = tasmin + (
    #             tas_day[(tas_day > t_tmin) & (tas_day < t_tmax)] - t_tmin) * delta_tas
    # tas_day[tas_day < t_tmin] = tasmin + (t_tmin - tas_day[tas_day < t_tmin]) * delta_tas

    tas_day[tas_day > t_tmax] = tasmax + (t_tmax - tas_day[tas_day > t_tmax]) * delta_tas * .5
    tas_day[(tas_day > t_tmin) & (tas_day < t_tmax)] = tasmin + (
            tas_day[(tas_day > t_tmin) & (tas_day < t_tmax)] - t_tmin) * delta_tas
    tas_day[tas_day < t_tmin] = tasmin + (t_tmin - tas_day[tas_day < t_tmin]) * delta_tas * .5
    # '''

    # sine curve ---
    # tas_day[tas_day > t_tmax] = (np.cos(
    #     (tas_day[tas_day > t_tmax] - t_tmin) * np.pi / t_tmax - t_tmin) + 1) * .5 * tasmin + (1 - (
    #             np.cos((tas_day[tas_day > t_tmax] - t_tmin) * np.pi / t_tmax - t_tmin) + 1) * .5) * tasmax
    #
    # tas_day[(tas_day > t_tmin) & (tas_day < t_tmax)] = (np.cos((24 + t_tmin + tas_day[
    #     (tas_day > t_tmin) & (tas_day < t_tmax)]) * np.pi / 24 + t_tmin - t_tmax) + 1) * .5 * tasmin + (1 - (np.cos(
    #     (24 + t_tmin + tas_day[(tas_day > t_tmin) & (tas_day < t_tmax)]) * np.pi / 24 + t_tmin - t_tmax) + 1) * .5) * tasmax
    #
    # tas_day[tas_day < t_tmin] = (np.cos(
    #     (t_tmin - tas_day[tas_day < t_tmin]) * np.pi / 24 + t_tmin - t_tmax) + 1) * .5 * tasmin + (1 - (
    #         np.cos(
    #             (t_tmin - tas_day[tas_day < t_tmin]) * np.pi / 24 + t_tmin - t_tmax) + 1) * .5) * tasmax

    '''
    # sine curve -------------------------------------------------------------------------------------------------------
    tas_day_con = tas_day.copy()

    cycle0 = (t_tmax - t_tmin) * 2
    cycle1 = (24 - (t_tmax - t_tmin)) * 2

    B0 = 2 * np.pi / cycle0
    B1 = 2 * np.pi / cycle1

    A0 = (tasmax - tasmin) / 2
    A1 = (tasmax - tasmin) / 2

    phase0 = cycle0 / 2
    phase1 = cycle1 / 2

    shift0 = (tasmin + tasmax) / 2
    shift1 = (tasmin + tasmax) / 2

    tas_day[tas_day_con > t_tmax] = A1 * np.cos(B1 * (tas_day_con[tas_day_con > t_tmax] - t_tmax)) + shift1
    tas_day[(tas_day_con > t_tmin) & (tas_day_con < t_tmax)] = A0 * np.cos(B0 * (tas_day_con[(tas_day_con > t_tmin) & (tas_day_con < t_tmax)] - t_tmin + phase0)) + shift0
    tas_day[tas_day_con < t_tmin] = A1 * np.cos(B1 * (tas_day_con[tas_day_con < t_tmin] - (cycle1 - 24) + cycle1 - t_tmax)) + shift1
    # '''

    # ----------------------------
    tas_day[t_tmin] = tasmin
    tas_day[t_tmax] = tasmax

    tas_arr = delta_T + tas_day[:, np.newaxis, np.newaxis]

    tss_arr = tas_arr.copy()
    tss_arr[tss_arr > 273.15] = 273.15

    return tas_arr, tss_arr, tas_day
