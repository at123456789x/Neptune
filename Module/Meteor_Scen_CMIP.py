# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Meteor_Scen_CMIP.py
@Time: 2023/12/1 19:42
@Function: 
"""

import os
from tqdm import trange
import pandas as pd
from glob import glob
import xarray as xr

def r_cmip(gcm_pth, gcm, ssp, dst_pth, cenlon, cenlat, rgi):

    df_dst = dst_pth + r'/CMIP_%s_%s.csv' % (gcm, ssp)

    if not os.path.exists(df_dst):

        # loop CMIP ---
        cmip_tmin_pth = sorted(glob(f'{gcm_pth}/tasmin/{gcm}/tasmin*{gcm}_{ssp}*.nc'))
        cmip_tmax_pth = sorted(glob(f'{gcm_pth}/tasmax/{gcm}/tasmax*{gcm}_{ssp}*.nc'))
        cmip_tas_pth = sorted(glob(f'{gcm_pth}/tas/{gcm}/tas*{gcm}_{ssp}*.nc'))
        cmip_pr_pth = sorted(glob(f'{gcm_pth}/pr/{gcm}/pr*{gcm}_{ssp}*.nc'))
        cmip_rh_pth = sorted(glob(f'{gcm_pth}/hurs/{gcm}/hurs*{gcm}_{ssp}*.nc'))
        cmip_uas_pth = sorted(glob(f'{gcm_pth}/uas/{gcm}/uas*{gcm}_{ssp}*.nc'))
        cmip_vas_pth = sorted(glob(f'{gcm_pth}/vas/{gcm}/vas*{gcm}_{ssp}*.nc'))

        for i in trange(len(cmip_tmin_pth), desc=f'R CMIP {gcm} {ssp} ({rgi})'):

            df_meteor_cmip_slice = pd.DataFrame()

            with xr.open_dataset(cmip_tmin_pth[i]) as ds_tmin:
                ds_tmin = ds_tmin.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(cmip_tmax_pth[i]) as ds_tmax:
                ds_tmax = ds_tmax.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(cmip_tas_pth[i]) as ds_tas:
                ds_tas = ds_tas.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(cmip_pr_pth[i]) as ds_pr:
                ds_pr = ds_pr.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(cmip_rh_pth[i]) as ds_rh:
                ds_rh = ds_rh.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(cmip_uas_pth[i]) as ds_uas:
                ds_uas = ds_uas.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(cmip_vas_pth[i]) as ds_vas:
                ds_vas = ds_vas.sel(lon=cenlon, lat=cenlat, method='Nearest')

            df_meteor_cmip_slice['tmin'] = ds_tmin['tasmin'].to_series()
            df_meteor_cmip_slice['tmax'] = ds_tmax['tasmax'].to_series()
            df_meteor_cmip_slice['tas'] = ds_tas['tas'].to_series()
            df_meteor_cmip_slice['pr'] = ds_pr['pr'].to_series() * 86400
            df_meteor_cmip_slice['rh'] = ds_rh['hurs'].to_series() * .01
            df_meteor_cmip_slice['uas'] = ds_uas['uas'].to_series()
            df_meteor_cmip_slice['vas'] = ds_vas['vas'].to_series()

            df_meteor_cmip = df_meteor_cmip_slice if i == 0 else pd.concat([df_meteor_cmip, df_meteor_cmip_slice])

        df_meteor_cmip.index = df_meteor_cmip.index.map(lambda x: pd.to_datetime(x.strftime('%Y-%m-%d %X')))

        df_meteor_cmip['wind'] = (df_meteor_cmip['uas'] ** 2 + df_meteor_cmip['vas'] ** 2) ** .5
        df_meteor_cmip.to_csv(df_dst)

    else:
        df_meteor_cmip = pd.read_csv(df_dst, index_col=0)
        df_meteor_cmip.index = pd.to_datetime(df_meteor_cmip.index)

    return df_meteor_cmip
