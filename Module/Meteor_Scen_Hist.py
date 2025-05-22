# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Meteor_Scen_Hist.py
@Time: 2023/12/1 19:30
@Function: 
"""

from tqdm import trange
import os
from glob import glob
import xarray as xr
import pandas as pd


def r_hist(gcm_pth, gcm, dst_pth, cenlon, cenlat, rgi):

    df_dst = dst_pth + r'/Hist_%s.csv' % gcm

    if not os.path.exists(df_dst):

        # glob pths ---
        hist_tmin_pth = sorted(glob(f'{gcm_pth}/tasmin/{gcm}/tasmin*{gcm}_historical*.nc'))
        hist_tmax_pth = sorted(glob(f'{gcm_pth}/tasmax/{gcm}/tasmax*{gcm}_historical*.nc'))
        hist_tas_pth = sorted(glob(f'{gcm_pth}/tas/{gcm}/tas*{gcm}_historical*.nc'))
        hist_pr_pth = sorted(glob(f'{gcm_pth}/pr/{gcm}/pr*{gcm}_historical*.nc'))
        hist_rh_pth = sorted(glob(f'{gcm_pth}/hurs/{gcm}/hurs*{gcm}_historical*.nc'))
        hist_uas_pth = sorted(glob(f'{gcm_pth}/uas/{gcm}/uas*{gcm}_historical*.nc'))
        hist_vas_pth = sorted(glob(f'{gcm_pth}/vas/{gcm}/vas*{gcm}_historical*.nc'))

        # loop hist ---
        for i in trange(len(hist_uas_pth), desc=f'R Hist {gcm} ({rgi})'):

            df_meteor_hist_slice = pd.DataFrame()

            with xr.open_dataset(hist_tmin_pth[i - len(hist_uas_pth)]) as ds_tmin:
                ds_tmin = ds_tmin.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(hist_tmax_pth[i - len(hist_uas_pth)]) as ds_tmax:
                ds_tmax = ds_tmax.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(hist_tas_pth[i - len(hist_uas_pth)]) as ds_tas:
                ds_tas = ds_tas.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(hist_pr_pth[i - len(hist_uas_pth)]) as ds_pr:
                ds_pr = ds_pr.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(hist_uas_pth[i - len(hist_uas_pth)]) as ds_uas:
                ds_uas = ds_uas.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(hist_vas_pth[i - len(hist_uas_pth)]) as ds_vas:
                ds_vas = ds_vas.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(hist_rh_pth[i - len(hist_uas_pth)]) as ds_rh:
                ds_rh = ds_rh.sel(lon=cenlon, lat=cenlat, method='Nearest')

            df_meteor_hist_slice['tmin'] = ds_tmin['tasmin'].to_series()
            df_meteor_hist_slice['tmax'] = ds_tmax['tasmax'].to_series()
            df_meteor_hist_slice['tas'] = ds_tas['tas'].to_series()
            df_meteor_hist_slice['pr'] = ds_pr['pr'].to_series() * 86400
            df_meteor_hist_slice['rh'] = ds_rh['hurs'].to_series() * .01
            df_meteor_hist_slice['uas'] = ds_uas['uas'].to_series()
            df_meteor_hist_slice['vas'] = ds_vas['vas'].to_series()

            df_meteor_hist = df_meteor_hist_slice if i == 0 else pd.concat(
                [df_meteor_hist, df_meteor_hist_slice])

        df_meteor_hist.index = df_meteor_hist.index.map(lambda x: pd.to_datetime(x.strftime('%Y-%m-%d %X')))

        df_meteor_hist['wind'] = (df_meteor_hist['uas'] ** 2 + df_meteor_hist['vas'] ** 2) ** .5

        df_meteor_hist.to_csv(df_dst)

    else:
        df_meteor_hist = pd.read_csv(df_dst, index_col=0)
        df_meteor_hist.index = pd.to_datetime(df_meteor_hist.index)

    return df_meteor_hist
