# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: wang.q@mail.hnust.edu.cn
@Software: PyCharm
@File: Resample.py
@Time: 2022/05/17 20:56
@Function: 
"""

import os
import multiprocessing
from osgeo import gdal, gdalconst
from glob import glob

def resampleTif2Tif(in_tif, ref_tif, out_tif):

    try:
        os.makedirs('\\'.join(out_tif.split('\\')[:-1]))
    except:
        pass

    # in_tif coord info
    in_ras = gdal.Open(in_tif, gdalconst.GA_ReadOnly)
    in_proj = in_ras.GetProjection()
    band_ref = in_ras.GetRasterBand(1)

    # ref_tif coord info
    ref_ras = gdal.Open(ref_tif, gdalconst.GA_ReadOnly)
    ref_proj = ref_ras.GetProjection()
    ref_trans = ref_ras.GetGeoTransform()
    x = ref_ras.RasterXSize
    y = ref_ras.RasterYSize

    # out_tif info
    driver = gdal.GetDriverByName('GTiff')
    out_ras = driver.Create(out_tif, x, y, 1, band_ref.DataType)
    # out_ras = driver.Create(out_tif, x, y, 1, gdalconst.GDT_Float32)
    out_ras.SetGeoTransform(ref_trans)
    out_ras.SetProjection(ref_proj)
    gdal.ReprojectImage(in_ras, out_ras, in_proj, ref_proj, gdalconst.GRA_Bilinear)

    # clc var
    del in_ras, ref_ras, out_ras

if __name__ == '__main__':

    ref_tif = r"E:\ProjectEtc\May_16th_NPP\NPP\MODISNPP0118\modisnpp2001_proj.tif"  # ref tif path
    output_folder = r'E:'  # output tif folder

    in_tif_path = r'F:\ERA5_App\Elr_Ann'  # in tifs folder
    in_tifs = glob(in_tif_path + r'\**\*.tif', recursive=True)

    Pool = multiprocessing.Pool(4)
    Pool.map(resampleTif2Tif, in_tifs)
