# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Calc_Solar_Rad.py
@Time: 2023/12/1 20:44
@Function: 
"""

# import matplotlib.pyplot as plt
import os
import solarpy as sp
import numpy as np
import pandas as pd

# Span Normal-Plane solar rad ---
# ~4 W m-2 per 100 m height rise (3.86)
def calc_solar_rad(dem_arr, dt, cenlat, dif_ratio, slope, aspect, f_sv, albedo_arr, dst_pth):

    dt_leap_year = dt.is_leap_year
    leap = 'leap' if dt_leap_year else 'noleap'
    date = dt.strftime('%m-%d')

    if not os.path.exists(f'{dst_pth}/DSR'):
        os.makedirs(f'{dst_pth}/DSR')

    date_dsr_pth = f'{dst_pth}/DSR/{date}_{leap}.npy'

    if os.path.exists(date_dsr_pth):
        solar_rad_down_day = np.load(date_dsr_pth)

    else:

        solar_rad_down_day = np.tile(dem_arr, (24, 1, 1)).astype(np.float16)

        z_span = 100
        heights = range(int(solar_rad_down_day[10][solar_rad_down_day[10] > -1e3].min() / z_span) * z_span,
                        (int(solar_rad_down_day[10].max() / z_span) + 1) * z_span, 10)
        solar_rad_span = [[sp.beam_irradiance(z + .5 * z_span, i, cenlat) for i in
                           pd.date_range(dt.to_timestamp(), periods=24, freq='1h')] for z in heights]

        for i in range(len(heights) - 1):
            height_mask = (heights[i] <= dem_arr) & (dem_arr <= heights[i + 1])
            solar_rad_down_day[np.tile(height_mask, (solar_rad_down_day.shape[0], 1, 1))] = np.repeat(
                solar_rad_span[i],
                np.sum(height_mask))

        np.save(date_dsr_pth, solar_rad_down_day)

    # split solar
    # R_dif = 0.15
    R_dif = 1 - dif_ratio[dt.month - 1]

    # solar direct rad
    date_azimuth_pth = f'{dst_pth}/DSR/{date}_{leap}_Azimuth.npy'
    date_zenith_pth = f'{dst_pth}/DSR/{date}_{leap}_Zenith.npy'

    if os.path.exists(date_azimuth_pth) and os.path.exists(date_zenith_pth):
        azimuth = np.load(date_azimuth_pth)
        zenith = np.load(date_zenith_pth)

    else:
        azimuth = np.array([sp.solar_azimuth(i, cenlat) for i in pd.date_range(dt.start_time, periods=24, freq='1h')])
        zenith = np.array([sp.theta_z(i, cenlat) for i in pd.date_range(dt.start_time, periods=24, freq='1h')])

        np.save(date_azimuth_pth, azimuth)
        np.save(date_zenith_pth, zenith)

    solar_rad_down_beam = solar_rad_down_day.copy()
    solar_rad_down_day = solar_rad_down_day * np.cos(zenith.reshape(-1, 1, 1))

    # solar_dir = (1 - R_dif) * solar_rad_down_day
    solar_dif = R_dif * solar_rad_down_day

    # plt.plot(solar_rad_down_day[:, 100, 100])
    # plt.plot(solar_dir[:, 100, 100])
    # plt.plot(solar_dif[:, 100, 100])
    # plt.show()

    solar_dir_shadow = solar_rad_down_beam * (1 - R_dif) * (
                np.cos(slope[np.newaxis, :, :]) * np.cos(zenith.reshape(-1, 1, 1)) +
                np.sin(slope[np.newaxis, :, :]) * np.sin(zenith.reshape(-1, 1, 1)) *
                np.cos(azimuth.reshape(-1, 1, 1) - aspect[np.newaxis, :, :]))

    solar_dir_shadow[solar_dir_shadow < 0] = 0

    # solar diffuse
    # solar_dif_sv = solar_dif * f_sv[np.newaxis, :, :]
    # solar_ter_sv = solar_rad_down_day * (1 - f_sv[np.newaxis, :, :]) * albedo
    solar_dif_terrain = solar_dif * f_sv[np.newaxis, :, :] + solar_rad_down_day * (
                1 - f_sv[np.newaxis, :, :]) * albedo_arr[np.newaxis, :, :]

    solar_dif_terrain[solar_dif_terrain < 0] = 0

    # solar tot rad
    solar_rad_down_tot = solar_dir_shadow + solar_dif_terrain

    return solar_rad_down_tot
