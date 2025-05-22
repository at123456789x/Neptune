# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Calc_Longwave_Rad.py
@Time: 2023/12/1 20:54
@Function: 
"""

import numpy as np
import pandas as pd

def calc_lw_rad(tas_arr, tss_arr, f_sv):

    stefan_const = 5.67e-8
    terrain_emission = 1  # debris is 0.96

    longwave_rad_down_day = stefan_const * tas_arr ** 4
    # longwave_rad_down_day_sky = 5.31e-13 / stefan_const * tas_arr ** 2 * longwave_rad_down_day * f_sv[np.newaxis, :, :]
    longwave_rad_down_day_sky = 9.365e-6 * tas_arr ** 2 * longwave_rad_down_day * f_sv[np.newaxis, :, :]
    longwave_rad_down_day_ter = longwave_rad_down_day * f_sv[np.newaxis, :, :] * (1 - terrain_emission)

    longwave_rad_down_day_ter_reflection = tas_arr.copy()
    longwave_rad_down_day_ter_reflection[longwave_rad_down_day_ter_reflection > 273.15] = 273.15
    longwave_rad_down_day_ter_reflection = (stefan_const * longwave_rad_down_day_ter_reflection ** 4 *
                                            (1 - f_sv[np.newaxis, :, :]) * terrain_emission)

    longwave_rad_down_day_ter = longwave_rad_down_day_ter + longwave_rad_down_day_ter_reflection

    longwave_rad_down_day = longwave_rad_down_day_sky + longwave_rad_down_day_ter

    longwave_rad_up_day = tss_arr.copy() * terrain_emission  # debris 0.96, snow & ice 1
    longwave_rad_up_day = stefan_const * longwave_rad_up_day ** 4

    longwave_rad_up_day = longwave_rad_up_day + (1 - terrain_emission) * (
                longwave_rad_down_day_sky + longwave_rad_down_day_ter)

    LR_net = longwave_rad_down_day - longwave_rad_up_day

    return LR_net, longwave_rad_down_day, longwave_rad_up_day
