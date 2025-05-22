# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: BC_Downscale.py
@Time: 2023/12/1 19:48
@Function:
"""

import numpy as np
import pandas as pd
from statsmodels.tsa import seasonal

def bias_correct(obs_ser, ref_ser, hist_ser, cmip_ser, method, sample):

    # df_bc_ref = pd.DataFrame(index=pd.date_range('1950-01-01', '2014-12-31', freq='M'))
    df_bc_ref = pd.DataFrame(index=pd.date_range('2000-01-01', '2014-12-31', freq='M'))

    if sample == 'mean':

        df_bc_ref['obs'] = obs_ser.resample('M').mean()
        df_bc_ref['hist'] = ref_ser.resample('M').mean()

    else:

        df_bc_ref['obs'] = obs_ser.resample('M').sum()
        df_bc_ref['hist'] = ref_ser.resample('M').sum()

    if method == '+':

        df_bc_exp = pd.concat([hist_ser, cmip_ser])

        df_bc_exp_hist = hist_ser.copy()
        df_bc_exp_cmip = cmip_ser.copy()

        df_bc_exp_hist_copy = df_bc_exp_hist[
            ~((df_bc_exp_hist.index.month == 2) & (df_bc_exp_hist.index.day == 29))].copy()
        df_bc_exp_cmip_copy = df_bc_exp_cmip[
            ~((df_bc_exp_cmip.index.month == 2) & (df_bc_exp_cmip.index.day == 29))].copy()

        decompose_obs = seasonal.seasonal_decompose(df_bc_ref['obs'], model='additive', period=12,
                                                    extrapolate_trend='freq')
        decompose_hist = seasonal.seasonal_decompose(df_bc_ref['hist'], model='additive', period=12,
                                                     extrapolate_trend='freq')
        decompose_exp_hist = seasonal.seasonal_decompose(df_bc_exp_hist_copy, model='additive', period=365,
                                                         extrapolate_trend='freq')
        decompose_exp_cmip = seasonal.seasonal_decompose(df_bc_exp_cmip_copy, model='additive', period=365,
                                                         extrapolate_trend='freq')

        lt_ratio = decompose_obs.trend.mean() - decompose_hist.trend.mean()
        season_ratio = decompose_obs.seasonal.std() / decompose_hist.seasonal.std()
        resid_ratio = decompose_obs.resid.std() / decompose_hist.resid.std()

        mvt_hist = (decompose_exp_hist.trend + lt_ratio + decompose_exp_hist.seasonal * season_ratio +
                    decompose_exp_hist.resid * resid_ratio)
        mvt_cmip = (decompose_exp_cmip.trend + lt_ratio + decompose_exp_cmip.seasonal * season_ratio +
                    decompose_exp_cmip.resid * resid_ratio)
        mvt = pd.concat([mvt_hist, mvt_cmip])

    elif method == '*':

        df_bc_ref[df_bc_ref <= 0] = 1e-5

        df_bc_exp = pd.concat([hist_ser, cmip_ser])
        df_bc_exp_copy = df_bc_exp[~((df_bc_exp.index.month == 2) & (df_bc_exp.index.day == 29))].copy()

        decompose_obs = seasonal.seasonal_decompose(df_bc_ref['obs'], model='multiplicative', period=12,
                                                    extrapolate_trend='freq')
        decompose_hist = seasonal.seasonal_decompose(df_bc_ref['hist'], model='multiplicative', period=12,
                                                     extrapolate_trend='freq')

        df_bc_exp_copy[df_bc_exp_copy <= 0] = 1e-5
        decompose_cmip = seasonal.seasonal_decompose(df_bc_exp_copy, model='multiplicative', period=365,
                                                     extrapolate_trend='freq')

        lt_ratio = decompose_obs.trend.mean() / decompose_hist.trend.mean()
        # lt_ratio = decompose_obs.trend / decompose_hist.trend
        # lt_ratio = lt_ratio.groupby(df_bc_ref.index.month).median()

        season_ratio = (decompose_obs.seasonal / decompose_hist.seasonal).mean()
        # season_ratio = decompose_obs.seasonal / decompose_hist.seasonal
        # season_ratio = season_ratio.groupby(df_bc_ref.index.month).median()

        # resid_ratio = (decompose_obs.resid / decompose_hist.resid).mean()
        resid_ratio = (decompose_obs.resid / decompose_hist.resid).median()
        # resid_ratio = decompose_obs.resid / decompose_hist.resid
        # resid_ratio = resid_ratio.groupby(df_bc_ref.index.month).median()

        # month bc
        # decompose_df = pd.DataFrame(index=df_bc_exp.index)

        # for i in range(12):
        #     decompose_df.loc[decompose_df.index.month == i + 1, 'lt'] = lt_ratio[i + 1]
        #     decompose_df.loc[decompose_df.index.month == i + 1, 'season'] = season_ratio[i + 1]
        #     decompose_df.loc[decompose_df.index.month == i + 1, 'resid'] = resid_ratio[i + 1]

        # mvt = (decompose_cmip.trend * lt_ratio * decompose_cmip.seasonal * decompose_df['season'] *
        #        decompose_cmip.resid * decompose_df['resid'])

        mvt = (decompose_cmip.trend * lt_ratio * decompose_cmip.seasonal * season_ratio *
               decompose_cmip.resid * resid_ratio)

        mvt[df_bc_exp <= 0] = 0

    elif method == 'Q':

        if sample == 'mean':

            df_bc_exp = pd.concat([hist_ser, cmip_ser])
            interp = np.interp(df_bc_exp.resample('M').mean(), df_bc_ref['hist'].sort_values(),
                               df_bc_ref['obs'].sort_values())
            interp = pd.DataFrame(interp / df_bc_exp.resample('M').mean(),
                                  index=df_bc_exp.resample('M').mean().index)

        else:

            df_bc_exp = pd.concat([hist_ser, cmip_ser])
            interp = np.interp(df_bc_exp.resample('M').sum(), df_bc_ref['hist'].sort_values(),
                               df_bc_ref['obs'].sort_values())
            interp = pd.DataFrame(interp / df_bc_exp.resample('M').sum(),
                                  index=df_bc_exp.resample('M').sum().index)

        mvt = pd.DataFrame(df_bc_exp)

        for ind in interp.index:
            mvt[(mvt.index.month == ind.month) & (mvt.index.year == ind.year)] *= interp.loc[ind].values

    elif method == 'delta':

        df_bc_exp = pd.concat([hist_ser, cmip_ser])
        month_r = df_bc_ref['obs'].groupby(df_bc_ref.index.month).mean() / df_bc_ref['hist'].groupby(
            df_bc_ref.index.month).mean()

        mvt = pd.DataFrame(df_bc_exp)

        for ind in month_r.index:
            mvt[mvt.index.month == ind] *= month_r.loc[ind]

    elif method == 'delta_ann':

        df_bc_exp = pd.concat([hist_ser, cmip_ser])
        mvt = pd.DataFrame(df_bc_exp) * (df_bc_ref['obs'].resample('Y').sum().mean() / df_bc_ref['hist'].resample('Y').sum().mean())

    elif method == '+*':

        df_bc_exp = pd.concat([hist_ser, cmip_ser])

        df_bc_exp_hist = hist_ser.copy()
        df_bc_exp_cmip = cmip_ser.copy()

        df_bc_exp_hist_copy = df_bc_exp_hist[
            ~((df_bc_exp_hist.index.month == 2) & (df_bc_exp_hist.index.day == 29))].copy()
        df_bc_exp_cmip_copy = df_bc_exp_cmip[
            ~((df_bc_exp_cmip.index.month == 2) & (df_bc_exp_cmip.index.day == 29))].copy()

        decompose_obs = seasonal.seasonal_decompose(df_bc_ref['obs'], model='additive', period=12,
                                                    extrapolate_trend='freq')
        decompose_hist = seasonal.seasonal_decompose(df_bc_ref['hist'], model='additive', period=12,
                                                     extrapolate_trend='freq')
        decompose_exp_hist = seasonal.seasonal_decompose(df_bc_exp_hist_copy, model='additive', period=365,
                                                         extrapolate_trend='freq')
        decompose_exp_cmip = seasonal.seasonal_decompose(df_bc_exp_cmip_copy, model='additive', period=365,
                                                         extrapolate_trend='freq')

        lt_ratio = decompose_obs.trend.mean() / decompose_hist.trend.mean()
        # season_ratio = decompose_obs.seasonal.std() / decompose_hist.seasonal.std()
        season_ratio = (decompose_obs.seasonal / decompose_hist.seasonal).mean()
        # resid_ratio = decompose_obs.resid.std() / decompose_hist.resid.std()
        resid_ratio = (decompose_obs.resid / decompose_hist.resid).median()

        mvt_hist = (decompose_exp_hist.trend * lt_ratio + decompose_exp_hist.seasonal * season_ratio +
                    decompose_exp_hist.resid * resid_ratio)
        mvt_cmip = (decompose_exp_cmip.trend * lt_ratio + decompose_exp_cmip.seasonal * season_ratio +
                    decompose_exp_cmip.resid * resid_ratio)
        mvt = pd.concat([mvt_hist, mvt_cmip])

    # plt.figure(figsize=(12, 6))
    # plt.plot(mvt.index.year.unique(), mvt.resample('Y').mean().values, label='Pr BC', linestyle='--', linewidth=3)
    # plt.plot(df_bc_exp.index.year.unique(), df_bc_exp.resample('Y').mean().values, label='Origin', linewidth=3)
    # plt.plot(df_bc_ref['obs'].index.year.unique(), df_bc_ref['obs'].resample('Y').mean().values, label='Obs', c='r',
    #          linewidth=4, alpha=0.8)
    # plt.legend(frameon=False)
    # plt.xlim(1850, 2100)
    # # plt.ylim(0, 1100)
    # # plt.yticks(np.arange(258.15, 288.15, 5), np.arange(-15, 15, 5))
    # plt.show()

    # mvt366 = pd.DataFrame(index=df_bc_exp.index)
    mvt366 = pd.DataFrame(index=pd.date_range(df_bc_exp.index[0], df_bc_exp.index[-1], freq='D'))
    mvt366['CMIP_BC'] = mvt

    # if method == '*':
    #     mvt366[df_bc_exp <= 0] = 0

    mvt366 = mvt366.interpolate()

    return mvt366

def bc_downscale(df_meteor_obs, df_meteor_hist, df_meteor_cmip):

    df_meteor_cmip_bc = pd.DataFrame()

    # df_meteor_cmip_bc['tmin'] = bias_correct(df_meteor_obs['tmin'], df_meteor_hist['tmin'],
    #                                          df_meteor_hist['tmin'],
    #                                          df_meteor_cmip['tmin'], '+', 'mean')
    # df_meteor_cmip_bc['tmax'] = bias_correct(df_meteor_obs['tmax'], df_meteor_hist['tmax'],
    #                                          df_meteor_hist['tmax'],
    #                                          df_meteor_cmip['tmax'], '+', 'mean')
    df_meteor_cmip_bc['tmin'] = bias_correct(df_meteor_obs['t2m'], df_meteor_hist['tas'],
                                             df_meteor_hist['tmin'],
                                             df_meteor_cmip['tmin'], '+', 'mean')
    df_meteor_cmip_bc['tmax'] = bias_correct(df_meteor_obs['t2m'], df_meteor_hist['tas'],
                                             df_meteor_hist['tmax'],
                                             df_meteor_cmip['tmax'], '+', 'mean')
    df_meteor_cmip_bc['tas'] = bias_correct(df_meteor_obs['t2m'], df_meteor_hist['tas'], df_meteor_hist['tas'],
                                            df_meteor_cmip['tas'], '+', 'mean')
    df_meteor_cmip_bc['pr'] = bias_correct(df_meteor_obs['pr'], df_meteor_hist['pr'], df_meteor_hist['pr'],
                                           df_meteor_cmip['pr'], 'delta_ann', 'sum')
    df_meteor_cmip_bc['rh'] = bias_correct(df_meteor_obs['rh'], df_meteor_hist['rh'], df_meteor_hist['rh'],
                                           df_meteor_cmip['rh'], 'delta_ann', 'mean')
    df_meteor_cmip_bc['wind'] = bias_correct(df_meteor_obs['wind'], df_meteor_hist['wind'],
                                             df_meteor_hist['wind'],
                                             df_meteor_cmip['wind'], 'delta_ann', 'mean')

    df_meteor_cmip_bc['pr'][df_meteor_cmip_bc['pr'] < 0] = 0
    df_meteor_cmip_bc['rh'][df_meteor_cmip_bc['rh'] < 0] = 0
    df_meteor_cmip_bc['rh'][df_meteor_cmip_bc['rh'] > 1] = 1
    df_meteor_cmip_bc['wind'][df_meteor_cmip_bc['wind'] < 0] = 0

    return df_meteor_cmip_bc
