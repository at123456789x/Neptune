# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Meteor_Scen_Obs.py
@Time: 2023/12/1 22:13
@Function: 
"""

from Meteor_Obs import *
from BC_Downscale import *

# TPMFD ---
def r_scen_tpmfd(cenlon, cenlat, elr_mean, med_z, dst_pth):

    df_dst = dst_pth + r'/Scenario_TPMFD.csv'

    if not os.path.exists(df_dst):

        df_meteor_cmfd = pd.DataFrame()

        for yr in range(2000, 2023):
            for mon in range(1, 13):

                temp_ncs = sorted(glob(r'I:\TPMFD\temp\daily\%s\*%s%02d*' % (yr, yr, mon)))
                prec_ncs = sorted(glob(r'I:\TPMFD\prcp\daily\%s\*%s%02d*' % (yr, yr, mon)))
                shum_ncs = sorted(glob(r'I:\TPMFD\shum\daily\%s\*%s%02d*' % (yr, yr, mon)))
                wind_ncs = sorted(glob(r'I:\TPMFD\wind\daily\%s\*%s%02d*' % (yr, yr, mon)))

                for i in trange(len(temp_ncs), desc='R Daily  TPMFD (%s-%02d)' % (yr, mon)):

                    # with xr.open_dataset(temp_ncs[i]) as cmfd_temp:
                    #     cmfd_temp = cmfd_temp.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
                    with xr.open_dataset(prec_ncs[i]) as cmfd_prec:
                        cmfd_prec = cmfd_prec.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
                    with xr.open_dataset(shum_ncs[i]) as cmfd_shum:
                        cmfd_shum = cmfd_shum.sel(longitude=cenlon, latitude=cenlat, method='Nearest')
                    with xr.open_dataset(wind_ncs[i]) as cmfd_wind:
                        cmfd_wind = cmfd_wind.sel(longitude=cenlon, latitude=cenlat, method='Nearest')

                    time_ser = pd.to_datetime(temp_ncs[i].split('\\')[-1].split('_')[-1].split('.')[0])

                    # df_meteor_cmfd.loc[time_ser, 'tas'] = float(cmfd_temp['temp'].data)
                    df_meteor_cmfd.loc[time_ser, 'sp'] = float(cmfd_shum['shum'].data)
                    df_meteor_cmfd.loc[time_ser, 'pr'] = float(cmfd_prec['prcp'].data) * 24
                    df_meteor_cmfd.loc[time_ser, 'wind'] = float(cmfd_wind['wind'].data)

                temp_hrs_ncs = sorted(glob(r'I:\TPMFD\temp\hourly\%s\*%s%02d*' % (yr, yr, mon)))

                for i in trange(len(temp_hrs_ncs), desc='R Hourly TPMFD (%s-%02d)' % (yr, mon)):
                    with xr.open_dataset(temp_hrs_ncs[i]) as cmfd_temp:
                        cmfd_temp = cmfd_temp.sel(longitude=cenlon, latitude=cenlat, method='Nearest')

                    time_ser = pd.to_datetime(temp_ncs[i].split('\\')[-1].split('_')[-1].split('.')[0])

                    df_meteor_cmfd.loc[time_ser, 'tmin'] = float(cmfd_temp.resample(time='D').min()['temp'])
                    df_meteor_cmfd.loc[time_ser, 'tmax'] = float(cmfd_temp.resample(time='D').max()['temp'])
                    df_meteor_cmfd.loc[time_ser, 'tas'] = float(cmfd_temp.resample(time='D').mean()['temp'])

        # rh ---
        pres = 1012.9 / ((-elr_mean / 1000 * med_z / df_meteor_cmfd['tas'] + 1) ** 5.257)
        Es = 6.1078 * np.exp(17.2693882 * (df_meteor_cmfd['tas'] - 273.16) / (df_meteor_cmfd['tas'] - 35.86))
        qs = 0.622 * Es / (pres - 0.378 * Es)

        df_meteor_cmfd['rh'] = df_meteor_cmfd['sp'] / qs
        df_meteor_cmfd['rh'][df_meteor_cmfd['rh'] > 1] = 1

        df_meteor_cmip_bc = df_meteor_cmfd.copy()
        df_meteor_cmip_bc.index = pd.date_range('2000-01-01 12:00', '2022-12-31 12:00', freq='D')

        df_meteor_cmip_bc.to_csv(df_dst)

    else:
        df_meteor_cmip_bc = pd.read_csv(df_dst, index_col=0)
        df_meteor_cmip_bc.index = pd.to_datetime(df_meteor_cmip_bc.index)

    return df_meteor_cmip_bc

# CMFD ---
def r_scen_cmfd(cenlon, cenlat, elr_mean, med_z, dst_pth):

    df_dst = dst_pth + r'/Scenario_CMFD.csv'

    if not os.path.exists(df_dst):

        for i in trange(2000, 2019, desc='R CMFD'):

            df_meteor_cmfd_slice = pd.DataFrame()

            with xr.open_dataset(r"E:\CMFD\temp_ITPCAS-CMFD_V0106_B-01_01dy_010deg_%s01-%s12.nc" % (i, i)) as cmfd_temp:
                cmfd_temp = cmfd_temp.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(r"E:\CMFD\prec_ITPCAS-CMFD_V0106_B-01_01dy_010deg_%s01-%s12.nc" % (i, i)) as cmfd_prec:
                cmfd_prec = cmfd_prec.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(r"E:\CMFD\shum_ITPCAS-CMFD_V0106_B-01_01dy_010deg_%s01-%s12.nc" % (i, i)) as cmfd_shum:
                cmfd_shum = cmfd_shum.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(r"E:\CMFD\wind_ITPCAS-CMFD_V0106_B-01_01dy_010deg_%s01-%s12.nc" % (i, i)) as cmfd_wind:
                cmfd_wind = cmfd_wind.sel(lon=cenlon, lat=cenlat, method='Nearest')

            df_meteor_cmfd_slice['tas'] = cmfd_temp['temp'].to_series()
            df_meteor_cmfd_slice['sp'] = cmfd_shum['shum'].to_series()
            df_meteor_cmfd_slice['pr'] = cmfd_prec['prec'].to_series() * 24
            df_meteor_cmfd_slice['wind'] = cmfd_wind['wind'].to_series()

            # hrs ---
            ncs = sorted(glob(r"E:\CMFD\Temp\*_ITPCAS-*%d*.nc" % i))

            for nc in ncs:
                with xr.open_dataset(nc) as cmfd_3hr_temp_mon:
                    cmfd_3hr_temp_mon = cmfd_3hr_temp_mon.sel(lon=cenlon, lat=cenlat, method='Nearest')

                cmfd_3hr_temp = xr.concat([cmfd_3hr_temp, cmfd_3hr_temp_mon], dim='time') if nc != ncs[0] else cmfd_3hr_temp_mon

            cmfd_3hr_temp_min = cmfd_3hr_temp.resample(time='D').min()
            cmfd_3hr_temp_max = cmfd_3hr_temp.resample(time='D').max()

            cmfd_3hr_temp_min['time'] = cmfd_3hr_temp_min['time'] + pd.to_timedelta(10.5, unit='h')
            cmfd_3hr_temp_max['time'] = cmfd_3hr_temp_max['time'] + pd.to_timedelta(10.5, unit='h')

            df_meteor_cmfd_slice['tmin'] = cmfd_3hr_temp_min['temp'].to_series()
            df_meteor_cmfd_slice['tmax'] = cmfd_3hr_temp_max['temp'].to_series()

            df_meteor_cmip_bc = df_meteor_cmfd_slice if i == 2000 else pd.concat([df_meteor_cmip_bc, df_meteor_cmfd_slice])

        # rh ---
        pres = 1012.9 / ((-elr_mean / 1000 * med_z / df_meteor_cmip_bc['tas'] + 1) ** 5.257)
        Es = 6.1078 * np.exp(17.2693882 * (df_meteor_cmip_bc['tas'] - 273.16) / (df_meteor_cmip_bc['tas'] - 35.86))
        qs = 0.622 * Es / (pres - 0.378 * Es)

        df_meteor_cmip_bc['rh'] = df_meteor_cmip_bc['sp'] / qs
        df_meteor_cmip_bc['rh'][df_meteor_cmip_bc['rh'] > 1] = 1

        df_meteor_cmip_bc.index = pd.date_range('2000-01-01 12:00', '2018-12-31 12:00', freq='D')

        df_meteor_cmip_bc.to_csv(df_dst)

    else:
        df_meteor_cmip_bc = pd.read_csv(df_dst, index_col=0)
        df_meteor_cmip_bc.index = pd.to_datetime(df_meteor_cmip_bc.index)

    return df_meteor_cmip_bc

# CFMD / TPFMD ---
def r_scen_cmfd_tpmfd(cenlon, cenlat, elr_mean, med_z, dst_pth):

    df_dst = dst_pth + r'/Scenario_CMFD_TPMFD.csv'

    if not os.path.exists(df_dst):

        df_meteor_cmfd = pd.DataFrame()

        for yr in range(2000, 2019):

            df_meteor_cmfd_slice = pd.DataFrame()

            with xr.open_dataset(
                    r"E:\CMFD\temp_ITPCAS-CMFD_V0106_B-01_01dy_010deg_%s01-%s12.nc" % (yr, yr)) as cmfd_temp:
                cmfd_temp = cmfd_temp.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(
                    r"E:\CMFD\prec_ITPCAS-CMFD_V0106_B-01_01dy_010deg_%s01-%s12.nc" % (yr, yr)) as cmfd_prec:
                cmfd_prec = cmfd_prec.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(
                    r"E:\CMFD\shum_ITPCAS-CMFD_V0106_B-01_01dy_010deg_%s01-%s12.nc" % (yr, yr)) as cmfd_shum:
                cmfd_shum = cmfd_shum.sel(lon=cenlon, lat=cenlat, method='Nearest')
            with xr.open_dataset(
                    r"E:\CMFD\wind_ITPCAS-CMFD_V0106_B-01_01dy_010deg_%s01-%s12.nc" % (yr, yr)) as cmfd_wind:
                cmfd_wind = cmfd_wind.sel(lon=cenlon, lat=cenlat, method='Nearest')

            df_meteor_cmfd_slice['t2m'] = cmfd_temp['temp'].to_series()
            df_meteor_cmfd_slice['sp'] = cmfd_shum['shum'].to_series()
            df_meteor_cmfd_slice['pr'] = cmfd_prec['prec'].to_series() * 24
            df_meteor_cmfd_slice['wind'] = cmfd_wind['wind'].to_series()

            df_meteor_cmfd_slice.index = df_meteor_cmfd_slice.index.map(lambda x: x - pd.to_timedelta(10.5, 'h'))

            df_meteor_cmfd = pd.concat([df_meteor_cmfd, df_meteor_cmfd_slice])

            for mon in range(1, 13):

                temp_hrs_ncs = sorted(glob(r'I:\TPMFD\temp\hourly\%s\*%s%02d*' % (yr, yr, mon)))

                for i in trange(len(temp_hrs_ncs), desc='R Hourly TPMFD (%s-%02d)' % (yr, mon)):
                    with xr.open_dataset(temp_hrs_ncs[i]) as cmfd_temp:
                        cmfd_temp = cmfd_temp.sel(longitude=cenlon, latitude=cenlat, method='Nearest')

                    time_ser = pd.to_datetime(temp_hrs_ncs[i].split('\\')[-1].split('_')[-3].split('.')[0])

                    df_meteor_cmfd.loc[time_ser, 'tmin'] = float(cmfd_temp.resample(time='D').min()['temp'])
                    df_meteor_cmfd.loc[time_ser, 'tmax'] = float(cmfd_temp.resample(time='D').max()['temp'])
                    df_meteor_cmfd.loc[time_ser, 'tas'] = float(cmfd_temp.resample(time='D').mean()['temp'])

        # rh ---
        pres = 1012.9 / ((-elr_mean / 1000 * med_z / df_meteor_cmfd['t2m'] + 1) ** 5.257)
        Es = 6.1078 * np.exp(17.2693882 * (df_meteor_cmfd['t2m'] - 273.16) / (df_meteor_cmfd['t2m'] - 35.86))
        qs = 0.622 * Es / (pres - 0.378 * Es)

        df_meteor_cmfd['rh'] = df_meteor_cmfd['sp'] / qs
        df_meteor_cmfd['rh'][df_meteor_cmfd['rh'] > 1] = 1

        df_meteor_cmip_bc = df_meteor_cmfd.copy()
        df_meteor_cmip_bc.index = pd.date_range('2000-01-01 12:00', '2018-12-31 12:00', freq='D')

        df_meteor_cmip_bc.to_csv(df_dst)

    else:
        df_meteor_cmip_bc = pd.read_csv(df_dst, index_col=0)
        df_meteor_cmip_bc.index = pd.to_datetime(df_meteor_cmip_bc.index)

    return df_meteor_cmip_bc

# HAR ---
def r_scen_har(cenlon, cenlat, elr_mean, med_z, dst_pth):

    df_dst = dst_pth + r'/Scenario_HAR.csv'

    if not os.path.exists(df_dst):

        df_meteor_har = pd.DataFrame()

        for yr in trange(2000, 2023):

            df_meteor_har_slice = pd.DataFrame()

            with xr.open_dataset(r"/datanew/hejp/HARv2_d10km/HARv2_d10km_d_2d_t2_%s.hdf" % yr) as har_temp:

                # Convert lon, lat to x, y ---
                if yr == 2000:
                    d_CenPt = ((har_temp['lon'] - cenlon) ** 2 + (har_temp['lat'] - cenlat) ** 2) ** .5
                    cenPt = d_CenPt.where(d_CenPt == d_CenPt.min(), drop=True)
                    cenX, cenY = cenPt.coords['west_east'].data[0], cenPt.coords['south_north'].data[0]

                har_temp = har_temp.sel(west_east=cenX, south_north=cenY, method='Nearest')

            with xr.open_dataset(r"/datanew/hejp/HARv2_d10km/HARv2_d10km_d_2d_prcp_%s.hdf" % yr) as har_prec:
                har_prec = har_prec.sel(west_east=cenX, south_north=cenY, method='Nearest')

            with xr.open_dataset(r"/datanew/hejp/HARv2_d10km/HARv2_d10km_d_2d_q2_%s.hdf" % yr) as har_q2:
                har_q2 = har_q2.sel(west_east=cenX, south_north=cenY, method='Nearest')

            with xr.open_dataset(r"/datanew/hejp/HARv2_d10km/HARv2_d10km_d_2d_ws10_%s.hdf" % yr) as har_wind:
                har_wind = har_wind.sel(west_east=cenX, south_north=cenY, method='Nearest')

            df_meteor_har_slice['t2m'] = har_temp['t2'].to_series()
            df_meteor_har_slice['sp'] = har_q2['q2'].to_series()
            df_meteor_har_slice['pr'] = har_prec['prcp'].to_series() * 24
            df_meteor_har_slice['wind'] = har_wind['ws10'].to_series()

            # hr tmp to get min and max daily tmp ---
            temp_hrs_nc = r'/datanew/hejp/HARv2_d10km/HARv2_d10km_h_2d_t2_%s.hdf' % yr

            with xr.open_dataset(temp_hrs_nc) as har_temp:
                har_temp = har_temp.sel(west_east=cenX, south_north=cenY, method='Nearest')

            df_meteor_har_slice['tmin'] = har_temp.resample(time='D').min()['t2'].to_series()
            df_meteor_har_slice['tmax'] = har_temp.resample(time='D').max()['t2'].to_series()
            df_meteor_har_slice['tas'] = har_temp.resample(time='D').mean()['t2'].to_series()

            df_meteor_har = pd.concat([df_meteor_har, df_meteor_har_slice])

        # rh ---
        q = df_meteor_har['sp'] / (df_meteor_har['sp'] + 1)
        pres = 1012.9 / ((-elr_mean / 1000 * med_z / df_meteor_har['tas'] + 1) ** 5.257)
        Es = 6.1078 * np.exp(17.2693882 * (df_meteor_har['tas'] - 273.16) / (df_meteor_har['tas'] - 35.86))
        qs = 0.622 * Es / (pres - 0.378 * Es)

        df_meteor_har['rh'] = q / qs
        df_meteor_har['rh'][df_meteor_har['rh'] > 1] = 1

        df_meteor_cmip_bc = df_meteor_har.copy()
        df_meteor_cmip_bc.index = pd.date_range('2000-01-01 12:00', '2022-12-31 12:00', freq='D')

        df_meteor_cmip_bc.to_csv(df_dst)

    else:
        df_meteor_cmip_bc = pd.read_csv(df_dst, index_col=0)
        df_meteor_cmip_bc.index = pd.to_datetime(df_meteor_cmip_bc.index)

    return df_meteor_cmip_bc

# ERA5-CRU ---
def r_scen_cru_era5(cenlon, cenlat, elr_mean, med_z, dst_pth, rgi):

    df_dst = dst_pth + r'/Scenario_CRU_ERA5.csv'

    if not os.path.exists(df_dst):

        df_meteor_obs = r_cru_era5(cenlon, cenlat, elr_mean, med_z, dst_pth, rgi)

        df_meteor_era5 = pd.read_csv(f'K:/Proj_Monsoon/ERA5-Daily/ERA5-Land-Term-{rgi}.csv',
                                     index_col='date')
        df_meteor_era5.drop(['system:index', '.geo'], axis=1, inplace=True)
        df_meteor_era5.columns = ['dew', 'pr', 'tas', 'tmax', 'tmin', 'u', 'v']

        df_meteor_era5['pr'] = df_meteor_era5['pr'] * 1000
        df_meteor_era5['rh'] = (
                np.exp(17.2693882 * (df_meteor_era5['dew'] - 273.16) / (df_meteor_era5['dew'] - 35.86)) /
                np.exp(17.2693882 * (df_meteor_era5['t2m'] - 273.16) / (df_meteor_era5['t2m'] - 35.86)))
        df_meteor_era5['wind'] = (df_meteor_era5['u'] ** 2 + df_meteor_era5['v'] ** 2) ** .5

        df_meteor_era5.index = pd.to_datetime(df_meteor_era5.index)

        # BC ---
        df_meteor_cmip_bc = bc_downscale(df_meteor_obs, df_meteor_era5.loc[:'2014', :], df_meteor_era5.loc['2015':, :])
        df_meteor_cmip_bc.to_csv(df_dst)

    else:
        df_meteor_cmip_bc = pd.read_csv(df_dst, index_col=0)
        df_meteor_cmip_bc.index = pd.to_datetime(df_meteor_cmip_bc.index)

    return df_meteor_cmip_bc
