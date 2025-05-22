# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Reprojection_UTM.py
@Time: 2023/12/1 17:27
@Function: 
"""

import os
import rasterio as rio
import rasterio.mask as rmask
from Resample import resampleTif2Tif
from rasterio.warp import calculate_default_transform, reproject, Resampling


# r & re-projection raster to UTM zone ---
def re_projection_utm(rgi, dem_pth, cur_glc, src_pth, dst_pth, dst_crs):

    with rio.open(dem_pth) as dem:
        rmask_arr, rmask_transform = rmask.mask(dem, [cur_glc['buffer']], crop=True, filled=False)
        meta = dem.meta
        meta.update({'transform': rmask_transform,
                     'height': rmask_arr.shape[1],
                     'width': rmask_arr.shape[2]
                     })

    with rio.open(dst_pth + r'/dem_wgs84.tif', 'w', **meta) as dst:
        dst.write(rmask_arr.data)

    with rio.open(dst_pth + r'/dem_wgs84.tif') as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(dst_pth + r'/dem_utm.tif', 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )

    resampleTif2Tif(src_pth + r'/%s/thk.tif' % rgi, dst_pth + r'/dem_utm.tif', dst_pth + r'/thk_utm.tif')

    if os.path.exists(src_pth + r'/%s/debris.tif' % rgi):
        resampleTif2Tif(src_pth + r'/%s/debris.tif' % rgi, dst_pth + r'/dem_utm.tif', dst_pth + r'/debris_utm.tif')
