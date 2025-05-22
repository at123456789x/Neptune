# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Generate_Init_Field.py
@Time: 2023/12/1 21:32
@Function: 
"""

import numpy as np
import xarray as xr

def gener_init_filed(xs, ys, xss, yss, thk_arr, ts_init, dem_arr, dst_pth):

    scale_ds = xr.Dataset()

    scale_ds.coords['time'] = np.array([0])
    scale_ds.coords['x'] = xs[0]
    scale_ds.coords['y'] = ys[:, 0]

    scale_ds['lon'] = (('time', 'y', 'x'), xss[np.newaxis, :, :])
    scale_ds['lat'] = (('time', 'y', 'x'), yss[np.newaxis, :, :])

    scale_ds['no_model_mask'] = (('time', 'y', 'x'), np.where(thk_arr == 0, 1, 0)[np.newaxis, :, :])

    scale_ds['time'].attrs = {'calender': 'none',
                              'long_name': 'Time',
                              'calendar': '365_day',
                              'standard_name': 'time',
                              'units': f'months since %s-01' % ts_init[0].strftime('%Y-%m')}

    scale_ds['x'].attrs = {'long_name': 'Cartesian x-coordinate',
                           'standard_name': 'projection_x_coordinate',
                           'units': 'meters'}

    scale_ds['y'].attrs = {'long_name': 'Cartesian y-coordinate',
                           'standard_name': 'projection_y_coordinate',
                           'units': 'meters'}

    scale_ds['topg'] = (('time', 'y', 'x'), (dem_arr - thk_arr)[np.newaxis, :, :])
    scale_ds['thk'] = (('time', 'y', 'x'), thk_arr[np.newaxis, :, :])

    scale_ds['no_model_mask'].attrs = {'units': '',
                                       'flag_meanings': 'normal special_treatment',
                                       'long_name': 'mask: zeros (modeling domain) and ones (no-model buffer near grid edges)',
                                       'pism_intent': 'model_state',
                                       'flag_values': np.array([0, 1])}

    scale_ds['lat'].attrs = {'long_name': 'Latitude',
                             'standard_name': 'latitude',
                             'units': 'degreeN'}

    scale_ds['lon'].attrs = {'long_name': 'Longitude',
                             'standard_name': 'longitude',
                             'units': 'degreeE'}

    scale_ds['topg'].attrs = {'long_name': 'Bedrock Topography',
                              'standard_name': 'bedrock_altitude',
                              'units': 'meters'}

    scale_ds['thk'].attrs = {'long_name': 'Ice Thickness',
                             'standard_name': 'land_ice_thickness',
                             'units': 'meters'}

    scale_ds = scale_ds.sortby('y')
    scale_ds.to_netcdf(dst_pth + r'/Init_Field.nc')
