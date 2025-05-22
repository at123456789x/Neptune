# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Generate_Time_Series.py
@Time: 2023/12/1 21:45
@Function: 
"""

import xarray as xr
import pandas as pd

def gener_ts(ice_loss_ser, tas_mean_ser, ts_init, xs, ys, gcm, ssp, dst_pth):

    spatial_ds = xr.Dataset()

    spatial_ds['climatic_mass_balance'] = (('time', 'y', 'x'), ice_loss_ser)
    spatial_ds['ice_surface_temp'] = (('time', 'y', 'x'), tas_mean_ser)

    cf_ser = list(range(len(ts_init)))

    spatial_ds.coords['time'] = cf_ser
    spatial_ds.coords['x'] = xs[0]
    spatial_ds.coords['y'] = ys[:, 0]

    spatial_ds['time'].attrs = {'calender': 'none',
                                'long_name': 'Time',
                                'calendar': '365_day',
                                'standard_name': 'time',
                                'units': f'months since %s-01' % ts_init[0].strftime('%Y-%m')}

    spatial_ds['climatic_mass_balance'].attrs = {'long_name': 'Surface Mass Balance',
                                                 'standard_name': 'land_ice_surface_specific_mass_balance_flux',
                                                 'units': 'kg m-2 s-1'}

    spatial_ds['x'].attrs = {'long_name': 'Cartesian x-coordinate',
                             'standard_name': 'projection_x_coordinate',
                             'units': 'meters'}

    spatial_ds['y'].attrs = {'long_name': 'Cartesian y-coordinate',
                             'standard_name': 'projection_y_coordinate',
                             'units': 'meters'}

    spatial_ds['ice_surface_temp'].attrs = {'long_name': 'Annual Mean Air Temperature (2 meter)',
                                            'standard_name': 'air_temperature',
                                            'units': 'Kelvin'}

    # spatial_ds.attrs = {'proj': rio.crs.CRS.from_wkt(ds.spatial_ref.attrs['crs_wkt']).to_proj4()}
    spatial_ds = spatial_ds.sortby('y')

    # spatial_ds.to_netcdf(r'F:\Proj_EB\RGI60-13.33006\SSP245_%d.nc' % init)
    spatial_ds.to_netcdf(f"{dst_pth}/{gcm}_{ssp}.nc")

    return spatial_ds, f"{dst_pth}/{gcm}_{ssp}.nc"
