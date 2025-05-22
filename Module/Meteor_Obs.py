# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Meteor_Obs.py
@Time: 2023/12/1 19:20
@Function: 
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import trange
from glob import glob

def r_cmfd(cenlon, cenlat, elr_mean, med_z, dst_pth):

    df_dst = dst_pth + r'/Obs_CMFD.csv'

    if not os.path.exists(df_dst):

        df_meteor_obs = pd.DataFrame()

        with xr.open_dataset(r"/mnt/e/CMFD/Data_forcing_01mo_010deg/temp_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc") as era_temp:
            era_temp = era_temp.sel(lon=cenlon, lat=cenlat, method='Nearest')
        with xr.open_dataset(r"/mnt/e/CMFD/Data_forcing_01mo_010deg/prec_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc") as era_pr:
            era_pr = era_pr.sel(lon=cenlon, lat=cenlat, method='Nearest')
        with xr.open_dataset(r"/mnt/e/CMFD/Data_forcing_01mo_010deg/shum_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc") as era_sp:
            era_sp = era_sp.sel(lon=cenlon, lat=cenlat, method='Nearest')
        with xr.open_dataset(r"/mnt/e/CMFD/Data_forcing_01mo_010deg/wind_CMFD_V0106_B-01_01mo_010deg_197901-201812.nc") as era_wind:
            era_wind = era_wind.sel(lon=cenlon, lat=cenlat, method='Nearest')

        df_meteor_obs['t2m'] = era_temp['temp'].to_series()
        df_meteor_obs['sp'] = era_sp['shum'].to_series()
        df_meteor_obs['pr'] = era_pr['prec'].to_series() * era_pr['time'].dt.days_in_month * 24
        df_meteor_obs['wind'] = era_wind['wind'].to_series()

        # rh ---
        pres = 1012.9 / ((-elr_mean / 1000 * med_z / df_meteor_obs['t2m'] + 1) ** 5.257)
        Es = 6.1078 * np.exp(17.2693882 * (df_meteor_obs['t2m'] - 273.16) / (df_meteor_obs['t2m'] - 35.86))
        qs = 0.622 * Es / (pres - 0.378 * Es)

        df_meteor_obs['rh'] = df_meteor_obs['sp'] / qs
        df_meteor_obs.loc[df_meteor_obs['rh'] > 1, 'rh'] = 1

        df_meteor_obs.to_csv(df_dst)

    else:
        df_meteor_obs = pd.read_csv(df_dst, index_col=0)
        df_meteor_obs.index = pd.to_datetime(df_meteor_obs.index)

    return df_meteor_obs

def r_tpmfd(cenlon, cenlat, elr_mean, med_z, dst_pth):

    df_dst = dst_pth + r'/Obs_TPMFD.csv'

    if not os.path.exists(df_dst):

        df_meteor_obs = pd.DataFrame()

        for yr in trange(1979, 2023, desc='Loop yrs'):
            for mon in range(1, 13):

                with xr.open_dataset(glob(r"/mnt/i/TPMFD/temp/monthly/*%d%02d.nc" % (yr, mon))[0]) as cmfd_temp:
                    cmfd_temp = cmfd_temp.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
                with xr.open_dataset(glob(r"/mnt/i/TPMFD/prcp/monthly/*%d%02d.nc" % (yr, mon))[0]) as cmfd_prec:
                    cmfd_prec = cmfd_prec.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
                with xr.open_dataset(glob(r"/mnt/i/TPMFD/shum/monthly/*%d%02d.nc" % (yr, mon))[0]) as cmfd_shum:
                    cmfd_shum = cmfd_shum.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
                with xr.open_dataset(glob(r"/mnt/i/TPMFD/wind/monthly/*%d%02d.nc" % (yr, mon))[0]) as cmfd_wind:
                    cmfd_wind = cmfd_wind.sel(longitude=cenlon, latitude=cenlat, method='Nearest')

                time_ser = pd.to_datetime('%s-%02d' % (yr, mon))

                df_meteor_obs.loc[time_ser, 't2m'] = float(cmfd_temp['temp'].data)
                df_meteor_obs.loc[time_ser, 'sp'] = float(cmfd_shum['shum'].data)
                df_meteor_obs.loc[time_ser, 'pr'] = float(cmfd_prec['prcp'].data) * time_ser.days_in_month * 24
                df_meteor_obs.loc[time_ser, 'wind'] = float(cmfd_wind['wind'].data)

        # rh ---
        pres = 1012.9 / ((-elr_mean / 1000 * med_z / df_meteor_obs['t2m'] + 1) ** 5.257)
        Es = 6.1078 * np.exp(17.2693882 * (df_meteor_obs['t2m'] - 273.16) / (df_meteor_obs['t2m'] - 35.86))
        qs = 0.622 * Es / (pres - 0.378 * Es)

        df_meteor_obs['rh'] = df_meteor_obs['sp'] / qs
        # df_meteor_obs.loc[df_meteor_obs['rh'] > 1, 'rh'] = 1

        df_meteor_obs.to_csv(df_dst)

    else:
        df_meteor_obs = pd.read_csv(df_dst, index_col=0)
        df_meteor_obs.index = pd.to_datetime(df_meteor_obs.index)

    return df_meteor_obs

def r_era5(cenlon, cenlat, elr_mean, med_z, dst_pth):

    df_dst = dst_pth + r'/Obs_ERA5.csv'

    if not os.path.exists(df_dst):

        df_meteor_obs = pd.DataFrame()

        with xr.open_dataset(r'/mnt/e/ERA5/Monthly/t2m.nc') as era_temp:
            era_temp = era_temp.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
        with xr.open_dataset(r'/mnt/e/ERA5/Monthly/pr.nc') as era_pr:
            era_pr = era_pr.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
        with xr.open_dataset(r'/mnt/e/ERA5/Monthly/dew_temp.nc') as era_dew:
            era_dew = era_dew.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
        with xr.open_dataset(r'/mnt/e/ERA5/Monthly/uas.nc') as era_uas:
            era_uas = era_uas.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
        with xr.open_dataset(r'/mnt/e/ERA5/Monthly/vas.nc') as era_vas:
            era_vas = era_vas.sel(longitude=cenlon, latitude=cenlat, method='Nearest')

        df_meteor_obs['t2m'] = era_temp['t2m'].to_series()
        df_meteor_obs['dew'] = era_dew['d2m'].to_series()
        df_meteor_obs['pr'] = era_pr['tp'].to_series() * era_pr['time'].dt.days_in_month * 1e3
        df_meteor_obs['rh'] = (np.exp(17.2693882 * (df_meteor_obs['dew'] - 273.16) / (df_meteor_obs['dew'] - 35.86)) /
                               np.exp(17.2693882 * (df_meteor_obs['t2m'] - 273.16) / (df_meteor_obs['t2m'] - 35.86)))
        df_meteor_obs['uas'] = era_uas['u10'].to_series()
        df_meteor_obs['vas'] = era_vas['v10'].to_series()
        df_meteor_obs['wind'] = (df_meteor_obs['uas'] ** 2 + df_meteor_obs['vas'] ** 2) ** .5

        df_meteor_obs.to_csv(df_dst)

    else:
        df_meteor_obs = pd.read_csv(df_dst, index_col=0)
        df_meteor_obs.index = pd.to_datetime(df_meteor_obs.index)

    return df_meteor_obs

def r_cru_tpmfd(cenlon, cenlat, elr_mean, med_z, dst_pth):

    df_dst = dst_pth + r'/Obs_CRU_TPMFD.csv'

    if not os.path.exists(df_dst):

        df_meteor_obs = pd.DataFrame()

        with xr.open_dataset(r"/mnt/e/CRU/cru_ts4.07.1901.2022.pre.dat.nc") as ds:
            ds_pt = ds.sel(lon=cenlon, lat=cenlat, method='nearest')
            df_pre = ds_pt['pre'].to_series()

        with xr.open_dataset(r"/mnt/e/CRU/cru_ts4.07.1901.2022.tmp.dat.nc") as ds:
            ds_pt = ds.sel(lon=cenlon, lat=cenlat, method='nearest')
            df_tmp = ds_pt['tmp'].to_series()

        with xr.open_dataset(r"/mnt/e/CRU/cru_ts4.07.1901.2022.vap.dat.nc") as ds:
            ds_pt = ds.sel(lon=cenlon, lat=cenlat, method='nearest')
            df_vap = ds_pt['vap'].to_series()

        df_rhm = df_vap / (6.1078 * np.exp(17.2693882 * (df_tmp + 273.15 - 273.16) / (df_tmp + 273.15 - 35.86)))

        df_meteor_obs['pr'] = df_pre
        df_meteor_obs['rh'] = df_rhm

        df_meteor_obs.index = df_meteor_obs.index.map(lambda x: pd.to_datetime(str(x.year) + '-' + str(x.month)))

        for yr in trange(1979, 2023, desc='Loop yrs'):
            for mon in range(1, 13):
                with xr.open_dataset(glob(r"/mnt/i/TPMFD/temp/monthly/*%d%02d.nc" % (yr, mon))[0]) as cmfd_temp:
                    cmfd_temp = cmfd_temp.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
                with xr.open_dataset(glob(r"/mnt/i/TPMFD/wind/monthly/*%d%02d.nc" % (yr, mon))[0]) as cmfd_wind:
                    cmfd_wind = cmfd_wind.sel(longitude=cenlon, latitude=cenlat, method='Nearest')

                time_ser = pd.to_datetime('%s-%02d' % (yr, mon))

                df_meteor_obs.loc[time_ser, 't2m'] = float(cmfd_temp['temp'].data)
                df_meteor_obs.loc[time_ser, 'wind'] = float(cmfd_wind['wind'].data)

        # rh ---
        df_meteor_obs.loc[df_meteor_obs['rh'] > 1, 'rh'] = 1
        df_meteor_obs.dropna(inplace=True)

        df_meteor_obs.to_csv(df_dst)

    else:
        df_meteor_obs = pd.read_csv(df_dst, index_col=0)
        df_meteor_obs.index = pd.to_datetime(df_meteor_obs.index)

    return df_meteor_obs

def r_har(cenlon, cenlat, elr_mean, med_z, dst_pth, rgi):

    df_dst = dst_pth + r'/Obs_HAR.csv'

    if not os.path.exists(df_dst):

        df_meteor_obs = pd.DataFrame()

        for yr in trange(2000, 2023):

            df_meteor_obs_slice = pd.DataFrame()

            with xr.open_dataset(r"K:/HARv2_d10km/HARv2_d10km_m_2d_t2_%s.hdf" % yr, decode_times=False) as har_temp:

                # Convert lon, lat to x, y ---
                if yr == 2000:
                    d_CenPt = ((har_temp['lon'] - cenlon) ** 2 + (har_temp['lat'] - cenlat) ** 2) ** .5
                    cenPt = d_CenPt.where(d_CenPt == d_CenPt.min(), drop=True)
                    cenX, cenY = cenPt.coords['west_east'].data[0], cenPt.coords['south_north'].data[0]

                har_temp.coords['time'] = pd.date_range(f'{yr}-01-01', f'{yr}-12-01', freq='MS')
                har_temp = har_temp.sel(west_east=cenX, south_north=cenY, method='Nearest')

            with xr.open_dataset(r"K:/HARv2_d10km/HARv2_d10km_m_2d_prcp_%s.hdf" % yr, decode_times=False) as har_prec:
                har_prec = har_prec.sel(west_east=cenX, south_north=cenY, method='Nearest')
                har_prec.coords['time'] = pd.date_range(f'{yr}-01-01', f'{yr}-12-01', freq='MS')
                har_prec['prcp'] *= har_prec.time.dt.daysinmonth

            with xr.open_dataset(r"K:/HARv2_d10km/HARv2_d10km_m_2d_q2_%s.hdf" % yr, decode_times=False) as har_q2:
                har_q2.coords['time'] = pd.date_range(f'{yr}-01-01', f'{yr}-12-01', freq='MS')
                har_q2 = har_q2.sel(west_east=cenX, south_north=cenY, method='Nearest')

            with xr.open_dataset(r"K:/HARv2_d10km/HARv2_d10km_m_2d_ws10_%s.hdf" % yr, decode_times=False) as har_wind:
                har_wind.coords['time'] = pd.date_range(f'{yr}-01-01', f'{yr}-12-01', freq='MS')
                har_wind = har_wind.sel(west_east=cenX, south_north=cenY, method='Nearest')

            df_meteor_obs_slice['t2m'] = har_temp['t2'].to_series()
            df_meteor_obs_slice['sp'] = har_q2['q2'].to_series()
            df_meteor_obs_slice['pr'] = har_prec['prcp'].to_series() * 24
            df_meteor_obs_slice['wind'] = har_wind['ws10'].to_series()

            df_meteor_obs = pd.concat([df_meteor_obs, df_meteor_obs_slice])

        # rh ---
        q = df_meteor_obs['sp'] / (df_meteor_obs['sp'] + 1)
        pres = 1012.9 / ((-elr_mean / 1000 * med_z / df_meteor_obs['t2m'] + 1) ** 5.257)
        Es = 6.1078 * np.exp(17.2693882 * (df_meteor_obs['t2m'] - 273.16) / (df_meteor_obs['t2m'] - 35.86))
        qs = 0.622 * Es / (pres - 0.378 * Es)

        df_meteor_obs['rh'] = q / qs
        df_meteor_obs['rh'][df_meteor_obs['rh'] > 1] = 1

        df_meteor_obs.to_csv(df_dst)

    else:
        df_meteor_obs = pd.read_csv(df_dst, index_col=0)
        df_meteor_obs.index = pd.to_datetime(df_meteor_obs.index)

    return df_meteor_obs


def r_cru_pr_rh_era5(cenlon, cenlat, elr_mean, med_z, dst_pth, rgi):

    df_dst = dst_pth + r'/Obs_CRU_Pr_RH_ERA5.csv'

    if not os.path.exists(df_dst):

        df_meteor_obs = pd.DataFrame()

        with xr.open_dataset(r"/datanew/hejp/CRU/cru_ts4.07.1901.2022.pre.dat.nc") as ds:
            ds_pt = ds.sel(lon=cenlon, lat=cenlat, method='nearest')
            df_pre = ds_pt['pre'].to_series()

        with xr.open_dataset(r"/datanew/hejp/CRU/cru_ts4.07.1901.2022.tmp.dat.nc") as ds:
            ds_pt = ds.sel(lon=cenlon, lat=cenlat, method='nearest')
            df_tmp = ds_pt['tmp'].to_series()

        with xr.open_dataset(r"/datanew/hejp/CRU/cru_ts4.07.1901.2022.vap.dat.nc") as ds:
            ds_pt = ds.sel(lon=cenlon, lat=cenlat, method='nearest')
            df_vap = ds_pt['vap'].to_series()

        df_rhm = df_vap / (6.1078 * np.exp(17.2693882 * (df_tmp + 273.15 - 273.16) / (df_tmp + 273.15 - 35.86)))

        df_meteor_obs['pr'] = df_pre

        # rh ---
        df_meteor_obs['rh'] = df_rhm
        df_meteor_obs.loc[df_meteor_obs['rh'] > 1, 'rh'] = 1

        df_meteor_obs.index = df_meteor_obs.index.map(lambda x: pd.to_datetime(str(x.year) + '-' + str(x.month)))

        # with xr.open_dataset(r'/datanew/hejp/ERA5/Monthly/t2m.nc') as era_temp:
        #     era_temp = era_temp.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
        # with xr.open_dataset(r'/datanew/hejp/ERA5/Monthly/uas.nc') as era_uas:
        #     era_uas = era_uas.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
        # with xr.open_dataset(r'/datanew/hejp/ERA5/Monthly/vas.nc') as era_vas:
        #     era_vas = era_vas.sel(longitude=cenlon, latitude=cenlat, method='Nearest')

        # df_meteor_obs['t2m'] = era_temp['t2m'].to_series()
        # df_meteor_obs['uas'] = era_uas['u10'].to_series()
        # df_meteor_obs['vas'] = era_vas['v10'].to_series()
        # df_meteor_obs['wind'] = (df_meteor_obs['uas'] ** 2 + df_meteor_obs['vas'] ** 2) ** .5

        # ERA5 D ---
        df_meteor_era5 = pd.read_csv(f'/datanew/hejp/proj_eb/HMA-RGI-Daily-Scale/ERA5-Land-Term-{rgi}.csv',
                                     index_col='date')
        df_meteor_era5.drop(['system:index', '.geo'], axis=1, inplace=True)
        df_meteor_era5.columns = ['dew', 'pr', 't2m', 'tmax', 'tmin', 'u', 'v']

        df_meteor_era5['pr'] = df_meteor_era5['pr'] * 1000
        df_meteor_era5['rh'] = (
                    np.exp(17.2693882 * (df_meteor_era5['dew'] - 273.16) / (df_meteor_era5['dew'] - 35.86)) /
                    np.exp(17.2693882 * (df_meteor_era5['t2m'] - 273.16) / (df_meteor_era5['t2m'] - 35.86)))
        df_meteor_era5['wind'] = (df_meteor_era5['u'] ** 2 + df_meteor_era5['v'] ** 2) ** .5

        df_meteor_era5.index = pd.to_datetime(df_meteor_era5.index)

        # ERA5 M ---
        df_meteor_era5_mon = pd.DataFrame()

        df_meteor_era5_mon['wind'] = df_meteor_era5['wind'].resample('M').mean()
        df_meteor_era5_mon['t2m'] = df_meteor_era5['t2m'].resample('M').mean()
        df_meteor_era5_mon['tmin'] = df_meteor_era5['tmin'].resample('M').mean()
        df_meteor_era5_mon['tmax'] = df_meteor_era5['tmax'].resample('M').mean()

        df_meteor_era5_mon.index = df_meteor_era5_mon.index.map(lambda x: pd.to_datetime('%s-%s-01' % (x.year, x.month)))

        df_meteor_obs = pd.concat([df_meteor_obs, df_meteor_era5_mon], axis=1)
        df_meteor_obs.dropna(inplace=True)

        df_meteor_obs.to_csv(df_dst)

    else:
        df_meteor_obs = pd.read_csv(df_dst, index_col=0)
        df_meteor_obs.index = pd.to_datetime(df_meteor_obs.index)

    return df_meteor_obs


def r_cru_pr_era5(cenlon, cenlat, elr_mean, med_z, dst_pth, rgi):

    df_dst = dst_pth + r'/Obs_CRU_Pr_ERA5.csv'

    if not os.path.exists(df_dst):

        df_meteor_obs = pd.DataFrame()

        with xr.open_dataset(r"E:/CRU/cru_ts4.07.1901.2022.pre.dat.nc") as ds:
            ds_pt = ds.sel(lon=cenlon, lat=cenlat, method='nearest')
            df_pre = ds_pt['pre'].to_series()

        with xr.open_dataset(r"E:/CRU/cru_ts4.07.1901.2022.tmp.dat.nc") as ds:
            ds_pt = ds.sel(lon=cenlon, lat=cenlat, method='nearest')
            df_tmp = ds_pt['tmp'].to_series()

        with xr.open_dataset(r"E:/CRU/cru_ts4.07.1901.2022.vap.dat.nc") as ds:
            ds_pt = ds.sel(lon=cenlon, lat=cenlat, method='nearest')
            df_vap = ds_pt['vap'].to_series()

        df_rhm = df_vap / (6.1078 * np.exp(17.2693882 * (df_tmp + 273.15 - 273.16) / (df_tmp + 273.15 - 35.86)))

        df_meteor_obs['pr'] = df_pre
        df_meteor_obs.index = df_meteor_obs.index.map(lambda x: pd.to_datetime(str(x.year) + '-' + str(x.month)))

        # with xr.open_dataset(r'/datanew/hejp/ERA5/Monthly/t2m.nc') as era_temp:
        #     era_temp = era_temp.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
        # with xr.open_dataset(r'/datanew/hejp/ERA5/Monthly/uas.nc') as era_uas:
        #     era_uas = era_uas.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
        # with xr.open_dataset(r'/datanew/hejp/ERA5/Monthly/vas.nc') as era_vas:
        #     era_vas = era_vas.sel(longitude=cenlon, latitude=cenlat, method='Nearest')

        # df_meteor_obs['t2m'] = era_temp['t2m'].to_series()
        # df_meteor_obs['uas'] = era_uas['u10'].to_series()
        # df_meteor_obs['vas'] = era_vas['v10'].to_series()
        # df_meteor_obs['wind'] = (df_meteor_obs['uas'] ** 2 + df_meteor_obs['vas'] ** 2) ** .5

        # ERA5 D ---
        df_meteor_era5 = pd.read_csv(f'K:/Proj_Monsoon/ERA5-Daily/ERA5-Land-Term-{rgi}.csv',
                                     index_col='date')
        df_meteor_era5.drop(['system:index', '.geo'], axis=1, inplace=True)
        df_meteor_era5.columns = ['dew', 'pr', 't2m', 'tmax', 'tmin', 'u', 'v']

        df_meteor_era5['pr'] = df_meteor_era5['pr'] * 1000
        df_meteor_era5['rh'] = (
                    np.exp(17.2693882 * (df_meteor_era5['dew'] - 273.16) / (df_meteor_era5['dew'] - 35.86)) /
                    np.exp(17.2693882 * (df_meteor_era5['t2m'] - 273.16) / (df_meteor_era5['t2m'] - 35.86)))
        df_meteor_era5['wind'] = (df_meteor_era5['u'] ** 2 + df_meteor_era5['v'] ** 2) ** .5

        df_meteor_era5.index = pd.to_datetime(df_meteor_era5.index)

        # ERA5 M ---
        df_meteor_era5_mon = pd.DataFrame()

        df_meteor_era5_mon['wind'] = df_meteor_era5['wind'].resample('M').mean()
        df_meteor_era5_mon['t2m'] = df_meteor_era5['t2m'].resample('M').mean()
        df_meteor_era5_mon['tmin'] = df_meteor_era5['tmin'].resample('M').mean()
        df_meteor_era5_mon['tmax'] = df_meteor_era5['tmax'].resample('M').mean()
        df_meteor_era5_mon['rh'] = df_meteor_era5['rh'].resample('M').mean()

        df_meteor_era5_mon.loc[df_meteor_era5_mon['rh'] > 1, 'rh'] = 1
        df_meteor_era5_mon.index = df_meteor_era5_mon.index.map(lambda x: pd.to_datetime('%s-%s-01' % (x.year, x.month)))

        df_meteor_obs = pd.concat([df_meteor_obs, df_meteor_era5_mon], axis=1)
        df_meteor_obs.dropna(inplace=True)

        df_meteor_obs.to_csv(df_dst)

    else:
        df_meteor_obs = pd.read_csv(df_dst, index_col=0)
        df_meteor_obs.index = pd.to_datetime(df_meteor_obs.index)

    return df_meteor_obs

