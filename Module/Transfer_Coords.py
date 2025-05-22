# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Transfer_Coords.py
@Time: 2023/12/2 0:07
@Function: 
"""

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import shapely

def gener_coords(xs, ys, dst_crs, dst_pth):

    if not os.path.exists(dst_pth + r'/Init_Field.nc'):

        if not os.path.exists(dst_pth + r'/coords.npz'):

            xss, yss = np.apply_along_axis(
                lambda x: np.concatenate(gpd.GeoSeries(shapely.geometry.Point(x), crs=dst_crs).to_crs('epsg:4326')[0].xy),
                0, np.array([xs, ys]))

            np.savez(dst_pth + r'/coords.npz', xss=xss, yss=yss)

        else:

            coords = np.load(dst_pth + r'/coords.npz')
            xss, yss = coords['xss'], coords['yss']

    else:
        with xr.open_dataset(dst_pth + r'/Init_Field.nc', decode_times=False) as ds:
            xss, yss = ds['lon'].data[0], ds['lat'].data[0][::-1, :]

    return xss, yss
