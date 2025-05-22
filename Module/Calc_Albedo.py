# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Calc_Albedo.py
@Time: 2023/12/1 20:27
@Function: 
"""

from Torch_Albedo import *


def calc_albedo(df_meteor_cmip_bc, dt, delta_T, wet_bulb, shift_d, pr_solid, dem_arr, debris_arr, thk_arr, xss, yss):

    df_albedo = pd.DataFrame()

    df_albedo['t2m'] = (df_meteor_cmip_bc.loc[dt, 'tas'] + delta_T - 273.15).flatten()
    # df_albedo['t2m'] = (df_meteor_cmip_bc.loc[dt + pd.to_timedelta(12, 'h'), 'tas'] + delta_T - 273.15).flatten()
    df_albedo['wet.bulb'] = wet_bulb[-1].flatten()

    for i in range(shift_d + 1):
        # df_albedo['pr.%d' % i] = df_meteor_cmip_bc.loc[dt - pd.to_timedelta(i, 'D') + pd.to_timedelta(12, 'h'), 'pr']
        df_albedo['pr.%d' % i] = df_meteor_cmip_bc.loc[dt - pd.to_timedelta(i, 'D'), 'pr']
        df_albedo['pr.sld.%d' % i] = pr_solid[shift_d - i].flatten()

    df_albedo['elevation'] = dem_arr.flatten()
    df_albedo['debris'] = debris_arr.flatten()
    df_albedo['lon'] = xss.flatten()
    df_albedo['lat'] = yss.flatten()
    df_albedo['mon'] = dt.month
    df_albedo['day'] = dt.day

    df_albedo['albedo'] = predict_albedo(df_albedo)

    albedo_arr = df_albedo['albedo'].to_numpy().reshape(dem_arr.shape[0], -1) / 100
    albedo_arr[albedo_arr < 0] = 0
    albedo_arr[albedo_arr > 1] = 1

    # albedo_arr[pr_solid[-3:].sum(axis=0) >= 2.5] = 0.9  # mid snow
    albedo_arr[albedo_arr < 0.1] = 0.1

    albedo_arr[thk_arr == 0] = np.nan

    # albedo_arr[pr_solid.sum(axis=0) > 0.1] += .2
    albedo_arr[albedo_arr > 1] = 1

    return albedo_arr


def calc_snow_albedo(snow_depth_arr, snow_days_arr, thk_arr, snow_depth_thres):

    snow_albedo_arr = 0.55 + (0.9 - 0.55) * np.exp(-snow_days_arr / 22)
    # snow_albedo_arr = 0.55 + (0.9 - 0.55) * np.exp(-snow_days_arr / 6)
    # snow_albedo_arr = snow_albedo_arr + (0.3 - snow_albedo_arr) * np.exp(-snow_depth_arr / 30)
    # snow_albedo_arr = snow_albedo_arr + (0.3 - snow_albedo_arr) * np.exp(-snow_depth_arr / 10)
    snow_albedo_arr = snow_albedo_arr + (0.3 - snow_albedo_arr) * np.exp(-snow_depth_arr / snow_depth_thres)
    snow_albedo_arr[thk_arr == 0] = np.nan

    return snow_albedo_arr
