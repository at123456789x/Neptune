# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Distinguish_Pr.py
@Time: 2023/12/1 20:14
@Function: 
"""

import numpy as np
import pandas as pd

def distinguish_pr(med_tas, t2m_arr, rh_arr, z_arr, tp_arr, dT_arr, dp, med_z):

    t2m_arr = t2m_arr + dT_arr - 273.15

    # tp_arr = np.tile(tp_arr, (1, z_arr.shape[1], z_arr.shape[2]))
    tp_arr = tp_arr * (1 + dp * (z_arr - med_z) / 100)

    E_s_med = 6.1078 * np.exp(17.2693882 * (med_tas - 273.16) / (med_tas - 35.86))
    E_med = E_s_med * rh_arr
    E_s = 6.1078 * np.exp(17.2693882 * (t2m_arr + 273.15 - 273.16) / (t2m_arr + 273.15 - 35.86))
    rh_arr = E_med / E_s
    rh_arr[rh_arr > 1] = 1
    rh_arr[rh_arr < 0] = 0

    t_wb_arr = (t2m_arr * np.arctan(0.151977 * (rh_arr * 100 + 8.313659) ** .5) +
                np.arctan(t2m_arr + rh_arr * 100) - np.arctan(rh_arr * 100 - 1.676331) +
                0.00391838 * (rh_arr * 100) ** 1.5 * np.arctan(0.023101 * rh_arr * 100) - 4.686035)

    # probability of precipitation type
    T0 = -5.87 - 0.1042 * z_arr / 1000 + 0.0885 * (z_arr / 1000) ** 2 + 16.06 * rh_arr - 9.614 * rh_arr ** 2
    # dT = 0.215 - 0.099 * np.tile(rh_arr, (1, z_arr.shape[1], z_arr.shape[2])) + 1.018 * rh_arr ** 2
    # dS = 2.374 - 1.634 * np.tile(rh_arr, (1, z_arr.shape[1], z_arr.shape[2]))
    dT = 0.215 - 0.099 * rh_arr + 1.018 * rh_arr ** 2
    dS = 2.374 - 1.634 * rh_arr

    # P1 = 1 / (1 + np.exp((t_wb_arr - T0 + dT) / dS))
    # P2 = 1 / (1 + np.exp((t_wb_arr - T0 - dT) / dS))

    Tmin = np.full_like(T0, np.nan)
    Tmax = np.full_like(T0, np.nan)

    prob_mask = (dT / dS) <= np.log(2)
    Tmin[prob_mask] = T0[prob_mask]
    Tmax[prob_mask] = T0[prob_mask]

    prob_mask = (dT / dS) > np.log(2)
    Tmin[prob_mask] = T0[prob_mask] - dS[prob_mask] * np.log(np.exp(dT[prob_mask] / dS[prob_mask]) -
                                                             2 * np.exp(-dT[prob_mask] / dS[prob_mask]))
    Tmax[prob_mask] = 2 * T0[prob_mask] - Tmin[prob_mask]

    Prob_sol = np.full_like(T0, np.nan)
    Prob_liq = np.full_like(T0, np.nan)

    Prob_sol[t_wb_arr >= Tmax] = 0
    Prob_liq[t_wb_arr >= Tmax] = tp_arr[t_wb_arr >= Tmax]

    Prob_sol[t_wb_arr <= Tmin] = tp_arr[t_wb_arr <= Tmin]
    Prob_liq[t_wb_arr <= Tmin] = 0

    prob_mask = (t_wb_arr < Tmax) & (t_wb_arr > Tmin)
    Prob_sol[prob_mask] = (tp_arr[prob_mask] *
                           (1 / (1 + np.exp((t_wb_arr[prob_mask] - T0[prob_mask]) / dS[prob_mask]))))
    Prob_liq[prob_mask] = (tp_arr[prob_mask] *
                           (1 - 1 / (1 + np.exp((t_wb_arr[prob_mask] - T0[prob_mask]) / dS[prob_mask]))))

    return t_wb_arr, Prob_sol


def calc_pr_type(wet_bulb, pr_solid, init_i, t_step, dt, ts_ser, df_meteor_cmip_bc, dem_arr, delta_T, shift_d, dp, med_z):

    # shift_d = 7 - 1

    if (dt == ts_ser[0]) & (init_i == 0) & (t_step == 0):

        # tas_mean = df_meteor_cmip_bc.loc[dt - pd.to_timedelta(shift_d, 'D') + pd.to_timedelta(12, 'h'):
        #                                  dt + pd.to_timedelta(12, 'h'), 'tas'].values
        # pr_total = df_meteor_cmip_bc.loc[dt - pd.to_timedelta(shift_d, 'D') + pd.to_timedelta(12, 'h'):
        #                                  dt + pd.to_timedelta(12, 'h'), 'pr'].values
        # rh_mean = df_meteor_cmip_bc.loc[dt - pd.to_timedelta(shift_d, 'D') + pd.to_timedelta(12, 'h'):
        #                                 dt + pd.to_timedelta(12, 'h'), 'rh'].values

        tas_mean = df_meteor_cmip_bc.loc[dt - pd.to_timedelta(shift_d, 'D'): dt, 'tas'].values
        pr_total = df_meteor_cmip_bc.loc[dt - pd.to_timedelta(shift_d, 'D'): dt, 'pr'].values
        rh_mean = df_meteor_cmip_bc.loc[dt - pd.to_timedelta(shift_d, 'D'): dt, 'rh'].values

        # tas_mean_dt = np.array([df_meteor_cmip_bc.loc[dt + pd.to_timedelta(12, 'h'), 'tas']])
        # pr_total_dt = np.array([df_meteor_cmip_bc.loc[dt + pd.to_timedelta(12, 'h'), 'pr']])
        # rh_mean_dt = np.array([df_meteor_cmip_bc.loc[dt + pd.to_timedelta(12, 'h'), 'rh']])

        tas_mean_dt = np.array([df_meteor_cmip_bc.loc[dt, 'tas']])
        pr_total_dt = np.array([df_meteor_cmip_bc.loc[dt, 'pr']])
        rh_mean_dt = np.array([df_meteor_cmip_bc.loc[dt, 'rh']])

        wet_bulb, pr_solid = distinguish_pr(tas_mean[:, np.newaxis, np.newaxis],
                                            tas_mean[:, np.newaxis, np.newaxis],
                                            rh_mean[:, np.newaxis, np.newaxis],
                                            dem_arr[np.newaxis, :, :],
                                            pr_total[:, np.newaxis, np.newaxis],
                                            delta_T[np.newaxis, :, :],
                                            dp,
                                            med_z)

    else:
        # tas_mean_dt = np.array([df_meteor_cmip_bc.loc[dt + pd.to_timedelta(12, 'h'), 'tas']])
        # pr_total_dt = np.array([df_meteor_cmip_bc.loc[dt + pd.to_timedelta(12, 'h'), 'pr']])
        # rh_mean_dt = np.array([df_meteor_cmip_bc.loc[dt + pd.to_timedelta(12, 'h'), 'rh']])

        tas_mean_dt = np.array([df_meteor_cmip_bc.loc[dt, 'tas']])
        pr_total_dt = np.array([df_meteor_cmip_bc.loc[dt, 'pr']])
        rh_mean_dt = np.array([df_meteor_cmip_bc.loc[dt, 'rh']])

        wet_bulb[:-1], pr_solid[:-1] = wet_bulb[1:], pr_solid[1:]
        wet_bulb[-1], pr_solid[-1] = distinguish_pr(tas_mean_dt[:, np.newaxis, np.newaxis],
                                                    tas_mean_dt[:, np.newaxis, np.newaxis],
                                                    rh_mean_dt[:, np.newaxis, np.newaxis],
                                                    dem_arr[np.newaxis, :, :],
                                                    pr_total_dt[:, np.newaxis, np.newaxis],
                                                    delta_T[np.newaxis, :, :],
                                                    dp,
                                                    med_z)

    return wet_bulb, pr_solid, tas_mean_dt, pr_total_dt, rh_mean_dt
