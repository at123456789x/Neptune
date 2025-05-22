# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Calc_Turbulence_Flux.py
@Time: 2023/12/1 21:01
@Function: 
"""

import numpy as np
import pandas as pd

def calc_tbl_flux(dem_arr, elr_mean, tas_arr, tss_arr, dt, df_meteor_cmip_bc, dw, med_z, med_tas):

    # T_arr = 15.04 - 0.00649 * dem_arr
    # pres_arr = 101.29 * (tas_arr / 288.08) ** 5.256
    pres_arr = 101.29 / ((-elr_mean / 1000 * dem_arr[np.newaxis, :, :] / tas_arr + 1) ** 5.257)
    # rh_value = df_meteor_cmip_bc.loc[dt + pd.to_timedelta(12, 'h'), 'rh']
    # wind_value = df_meteor_cmip_bc.loc[dt + pd.to_timedelta(12, 'h'), 'wind']

    rh_value = df_meteor_cmip_bc.loc[dt, 'rh']
    wind_value = df_meteor_cmip_bc.loc[dt, 'wind']

    E_s_med = 6.1078 * np.exp(17.2693882 * (med_tas - 273.16) / (med_tas - 35.86))
    E_med = E_s_med * rh_value
    E_s = 6.1078 * np.exp(17.2693882 * (tas_arr - 273.16) / (tas_arr - 35.86))
    rh_arr = E_med / E_s
    rh_arr[rh_arr > 1] = 1
    rh_arr[rh_arr < 0] = 0

    wind_arr = wind_value + ((dem_arr - med_z) * dw)

    # Sensible flux ---
    SH_arr = 1006 * 0.002 * (tas_arr - tss_arr) * (pres_arr / (.2869 * tas_arr)) * wind_arr

    # Latent flux ---
    # l_arr = (transfer_coef * 0.623 * 2.514e6 / (101325 / (dem_arr[np.newaxis, :, :] * (-elr_mean / 1000) / tas_arr + 1) ** 5.257 * 1005) *
    #          (6.1078 * np.exp(17.2693882 * (tas_arr - 273.16) / (tas_arr - 35.86)) * 100 - 611))

    Es_tas_arr = 6.1078 * np.exp(17.2693882 * (tas_arr - 273.16) / (tas_arr - 35.86))
    Es_tss_arr = 6.1078 * np.exp(17.2693882 * (tss_arr - 273.16) / (tss_arr - 35.86))

    qs_tas_arr = 0.622 * Es_tas_arr / (pres_arr * 10 - 0.378 * Es_tas_arr)
    qs_tss_arr = 0.622 * Es_tss_arr / (pres_arr * 10 - 0.378 * Es_tss_arr)

    # LH_arr = 2.514e6 * 0.002 * (pres_arr / (.2869 * tas_arr)) * wind_arr * (rh_value * qs_tas_arr - qs_tss_arr)
    LH_arr = 2.514e6 * 0.002 * (pres_arr / (.2869 * tas_arr)) * wind_arr * (rh_arr * qs_tas_arr - qs_tss_arr)

    return SH_arr, LH_arr
