# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: ELR_Parameters.py
@Time: 2023/12/1 17:52
@Function: 
"""

import xarray as xr
import numpy as np

# parameter ---
def elr_params(dem_pth, dem_arr, elr_pth, cenlon, cenlat):

    if 'HAR' in dem_pth:
        with xr.open_dataset(elr_pth) as ds_elr:
            with xr.open_dataset(dem_pth) as ds_dem:

                d_CenPt = ((ds_dem['lon'] - cenlon) ** 2 + (ds_dem['lat'] - cenlat) ** 2) ** .5
                cenPt = d_CenPt.where(d_CenPt == d_CenPt.min(), drop=True)
                cenX, cenY = cenPt.coords['west_east'].data[0], cenPt.coords['south_north'].data[0]

                med_z = ds_dem.sel(west_east=cenX, south_north=cenY, method='Nearest')['hgt'].data[0]

                delta_z = dem_arr - med_z
                elr_mean = ds_elr.sel(x=cenlon, y=cenlat, method='Nearest')['band_data'].data[0]
                delta_T = delta_z / 1000 * elr_mean
                delta_T[dem_arr == -9999] = np.nan

    elif 'ERA' in dem_pth:
        with xr.open_dataset(elr_pth) as ds_elr:
            with xr.open_dataset(dem_pth) as ds_dem:

                med_z = ds_dem.sel(longitude=cenlon, latitude=cenlat, method='Nearest')['z'].data[0] / 9.80665

                delta_z = dem_arr - med_z
                elr_mean = ds_elr.sel(x=cenlon, y=cenlat, method='Nearest')['band_data'].data[0]
                delta_T = delta_z / 1000 * elr_mean
                delta_T[dem_arr == -9999] = np.nan

    elif 'tpmfd' in dem_pth:
        with xr.open_dataset(elr_pth) as ds_elr:
            with xr.open_dataset(dem_pth) as ds_grid:

                lons = ds_grid['x'].data
                lats = ds_grid['y'].data
                lons, lats = np.meshgrid(lons, lats)
                distance = np.sqrt((lons - cenlon) ** 2 + (lats - cenlat) ** 2)
                ind_y, ind_x = np.argwhere(distance == distance.min())[0]
                cenlon, cenlat = lons[ind_y, ind_x], lats[ind_y, ind_x]
                med_z = ds_grid['band_data'].sel(y=cenlat, x=cenlon).data[0]

                delta_z = dem_arr - med_z
                elr_mean = ds_elr.sel(x=cenlon, y=cenlat, method='Nearest')['band_data'].data[0]
                delta_T = delta_z / 1000 * elr_mean
                delta_T[dem_arr == -9999] = np.nan

    elif 'CMFD_V0200' in dem_pth:
        with xr.open_dataset(elr_pth) as ds_elr:
            with xr.open_dataset(dem_pth) as ds_dem:

                med_z = ds_dem.sel(lon=cenlon, lat=cenlat, method='Nearest')['elev'].data[0]

                delta_z = dem_arr - med_z
                elr_mean = ds_elr.sel(x=cenlon, y=cenlat, method='Nearest')['band_data'].data[0]
                delta_T = delta_z / 1000 * elr_mean
                delta_T[dem_arr == -9999] = np.nan


    else:
        with xr.open_dataset(elr_pth) as ds_elr:
            with xr.open_dataset(dem_pth) as ds_dem:

                if dem_pth.endswith('tif'):
                    med_z = ds_dem.sel(x=cenlon, y=cenlat, method='Nearest')['band_data'].data[0]
                else:
                    med_z = ds_dem.sel(lon=cenlon, lat=cenlat, method='Nearest')['elev'].data[0]

                delta_z = dem_arr - med_z
                elr_mean = ds_elr.sel(x=cenlon, y=cenlat, method='Nearest')['band_data'].data[0]
                delta_T = delta_z / 1000 * elr_mean
                delta_T[dem_arr == -9999] = np.nan

    return med_z, elr_mean, delta_T
