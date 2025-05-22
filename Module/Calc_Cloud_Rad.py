    # !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Calc_Cloud_Rad.py
@Time: 2023/12/1 21:10
@Function: 
"""

import numpy as np
import pandas as pd


def calc_cloud_rad(rh_value, pr_total_dt, longwave_rad_down_day):

    if 0.7 <= rh_value < 0.8:
        cloud_fraction = (rh_value - 0.7) * 2 + 0.1
    elif 0.8 <= rh_value < 0.9:
        cloud_fraction = (rh_value - 0.8) * 4 + 0.3
    elif rh_value >= 0.9:
        cloud_fraction = (rh_value - 0.9) * 3 + 0.7
    else:
        cloud_fraction = rh_value / 0.7 * 0.1

    if 0.1 <= pr_total_dt < 1:
        cloud_fraction = max(0.8 + (pr_total_dt - 0.1) / 9, cloud_fraction)
    if 1 <= pr_total_dt < 5:
        cloud_fraction = max(0.9 + (pr_total_dt - 1) / 40, cloud_fraction)
    if pr_total_dt >= 5:
        cloud_fraction = 1

    cloud_albedo = 0.07 * np.e ** (1.93 * cloud_fraction)

    Cloud_net = longwave_rad_down_day * 0.2 * cloud_fraction

    return Cloud_net, cloud_fraction, cloud_albedo


# cloud diagnose ---
def calc_cloud_rad_diagnose(rh_value, pr_total_dt, longwave_rad_down_day):

    if 0.7 <= rh_value < 0.8:
        cloud_fraction = (rh_value - 0.7) * 2 + 0.1
    elif 0.8 <= rh_value < 0.9:
        cloud_fraction = (rh_value - 0.8) * 4 + 0.3
    elif rh_value >= 0.9:
        cloud_fraction = (rh_value - 0.9) * 3 + 0.7
    else:
        cloud_fraction = rh_value / 0.7 * 0.1

    if rh_value >= 0.8:
        cloud_albedo = 0.7
    elif 0.7 <= rh_value < 0.8:
        cloud_albedo = 0.5
    elif pr_total_dt[0] >= 0.1:
        cloud_albedo, cloud_fraction = 0.65, 0.8
    elif pr_total_dt[0] >= 10:
        cloud_albedo, cloud_fraction = 0.8, 1
    elif pr_total_dt[0] >= 25:
        cloud_albedo, cloud_fraction = 0.9, 1
    else:
        cloud_albedo = 0.3

    Cloud_net = longwave_rad_down_day * 0.2 * cloud_fraction

    return Cloud_net, cloud_fraction, cloud_albedo


# model of Jacobs, J.D., 1978. Radiation climate of broughton island.
def calc_cloud_rad_Jacobs(rh_value, longwave_rad_down_day, dem_arr, elr_mean, tas_arr, med_z, med_tas):

    # pres_arr = 101.29 / ((-elr_mean / 1000 * dem_arr[np.newaxis, :, :] / tas_arr.mean(axis=0) + 1) ** 5.257)
    # ref_pres = 101.29 / ((-elr_mean / 1000 * med_z / med_tas + 1) ** 5.257)

    # cloud_fraction_threshold = 0.7 + 0.2 * np.e ** (1 - (pres_arr / 100) ** 4)
    # cloud_fraction = 1 - np.sqrt(1 - ((rh_value - cloud_fraction_threshold) / (1 - cloud_fraction_threshold)))

    E_s_med = 6.1078 * np.exp(17.2693882 * (med_tas - 273.16) / (med_tas - 35.86))
    E_med = E_s_med * rh_value
    E_s = 6.1078 * np.exp(17.2693882 * (tas_arr - 273.16) / (tas_arr - 35.86))
    rh_arr = E_med / E_s
    rh_arr[rh_arr > 1] = 1
    rh_arr[rh_arr < 0] = 0

    df_rh_threshold = pd.DataFrame(np.array([[0, 3000, 8000, 10000], [0.95, 0.85, 0.99, 0.99]]).T, columns=['alt', 'rh_threshold'])
    cloud_fraction_threshold = np.interp(dem_arr, df_rh_threshold['alt'], df_rh_threshold['rh_threshold'])

    cloud_fraction = 1 - np.sqrt((1 - rh_arr) / (1 - cloud_fraction_threshold))
    cloud_fraction[cloud_fraction > 1] = 1
    cloud_fraction[cloud_fraction < 0] = 0

    # cloud_albedo = (0.07 * np.e ** (1.93 * cloud_fraction)).copy()

    cloud_albedo = 0.52 * cloud_fraction
    Cloud_net = (longwave_rad_down_day * 0.26 * cloud_fraction).copy()

    # cloud_fraction = cloud_fraction.mean(axis=0)
    # cloud_albedo = cloud_albedo.mean(axis=0)
    # Cloud_net = Cloud_net.mean(axis=0)

    return Cloud_net, cloud_fraction, cloud_albedo

def calc_cloud_rad_wrf(rh_value, longwave_rad_down_day, dem_arr, elr_mean, tas_arr, med_z, med_tas):

    E_s_med = 6.1078 * np.exp(17.2693882 * (med_tas - 273.16) / (med_tas - 35.86))
    E_med = E_s_med * rh_value
    E_s = 6.1078 * np.exp(17.2693882 * (tas_arr - 273.16) / (tas_arr - 35.86))
    rh_arr = E_med / E_s
    rh_arr[rh_arr > 1] = 1
    rh_arr[rh_arr < 0] = 0

    cloud_fraction = 4 * rh_arr - 3
    cloud_fraction[cloud_fraction > 1] = 1
    cloud_fraction[cloud_fraction < 0] = 0
    cloud_fraction[np.isnan(cloud_fraction)] = 0

    # cloud_albedo = (0.07 * np.e ** (1.93 * cloud_fraction)).copy()
    cloud_albedo = 0.52 * cloud_fraction
    Cloud_net = (longwave_rad_down_day * 0.26 * cloud_fraction).copy()

    # cloud_fraction = cloud_fraction.mean(axis=0)
    # cloud_albedo = cloud_albedo.mean(axis=0)
    # Cloud_net = Cloud_net.mean(axis=0)

    return Cloud_net, cloud_fraction, cloud_albedo


def calc_cloud_rad_Engström(rh_value, longwave_rad_down_day, dem_arr, elr_mean, tas_arr, med_z, med_tas):

    E_s_med = 6.1078 * np.exp(17.2693882 * (med_tas - 273.16) / (med_tas - 35.86))
    E_med = E_s_med * rh_value
    E_s = 6.1078 * np.exp(17.2693882 * (tas_arr - 273.16) / (tas_arr - 35.86))
    rh_arr = E_med / E_s
    rh_arr[rh_arr > 1] = 1
    rh_arr[rh_arr < 0] = 0

    cloud_fraction = rh_arr.copy()

    cloud_fraction[(0.7 <= rh_arr) & (rh_arr < 0.8)] = (cloud_fraction[(0.7 <= rh_arr) & (rh_arr < 0.8)] - 0.7) * 2 + .1
    cloud_fraction[(0.8 <= rh_arr) & (rh_arr < 0.9)] = (cloud_fraction[(0.8 <= rh_arr) & (rh_arr < 0.9)] - 0.8) * 4 + .3
    cloud_fraction[rh_arr >= 0.9] = (cloud_fraction[rh_arr >= 0.9] - 0.9) * 3 + .7
    cloud_fraction[rh_arr < 0.7] = cloud_fraction[rh_arr < 0.7] / 0.7 * 0.1

    cloud_fraction[cloud_fraction > 1] = 1
    cloud_fraction[cloud_fraction < 0] = 0
    cloud_fraction[np.isnan(cloud_fraction)] = 0

    cloud_albedo = (0.07 * np.e ** (1.93 * cloud_fraction)).copy()
    Cloud_net = (longwave_rad_down_day * 0.26 * cloud_fraction).copy()

    # cloud_fraction = cloud_fraction.mean(axis=0)
    # cloud_albedo = cloud_albedo.mean(axis=0)
    # Cloud_net = Cloud_net.mean(axis=0)

    return Cloud_net, cloud_fraction, cloud_albedo

def calc_cloud_rad_wrf_Jacobs(rh_value, longwave_rad_down_day, tas_arr, med_tas, pr_total_dt):

    E_s_med = 6.1078 * np.exp(17.2693882 * (med_tas - 273.16) / (med_tas - 35.86))
    E_med = E_s_med * rh_value
    E_s = 6.1078 * np.exp(17.2693882 * (tas_arr - 273.16) / (tas_arr - 35.86))
    rh_arr = E_med / E_s
    rh_arr[rh_arr > 1] = 1
    rh_arr[rh_arr < 0] = 0

    cloud_fraction = 4 * rh_arr - 3
    cloud_fraction[cloud_fraction > 1] = 1
    cloud_fraction[cloud_fraction < 0] = 0
    cloud_fraction[np.isnan(cloud_fraction)] = 0

    cloud_albedo = cloud_fraction.copy()

    cloud_fraction_x = [0, 0.7, 0.8, 1]
    cloud_albedo_y = [0, 0.3, 0.5, 0.7]

    cloud_albedo = np.interp(cloud_albedo, cloud_fraction_x, cloud_albedo_y)

    if pr_total_dt >= 25:
        cloud_albedo[cloud_fraction < 0.9] = 0.9
    elif pr_total_dt >= 10:
        cloud_albedo[cloud_fraction < 0.8] = 0.8
    elif pr_total_dt >= 1:
        cloud_albedo[cloud_fraction < 0.7] = 0.7

    cloud_albedo[cloud_albedo < 0] = 0
    cloud_albedo[cloud_albedo > 1] = 1

    cloud_albedo *= cloud_fraction

    Cloud_net = (longwave_rad_down_day * 0.26 * cloud_fraction).copy()

    # cloud_fraction = cloud_fraction.mean(axis=0)
    # cloud_albedo = cloud_albedo.mean(axis=0)
    # Cloud_net = Cloud_net.mean(axis=0)

    return Cloud_net, cloud_fraction, cloud_albedo
