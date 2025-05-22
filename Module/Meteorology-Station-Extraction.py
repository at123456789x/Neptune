# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Meteorology-Station-Extraction.py
@Time: 2023/12/6 17:02
@Function: 
"""

import os
from tqdm import trange
import numpy as np
import pandas as pd
import geopandas as gpd

# set env ---
# yr, mon = 2000, 1
shp_pth = r"G:\CMA\全国气象站日值数据\01说明文件\站点分布图\站点分布图2424个.shp"

# tmp ---
for yr in range(2000, 2018):

    # yr = 2018
    for mon in range(8, 13):

        df = pd.read_table(r"G:\CMA\全国气象站日值数据\05气温\SURF_CLI_CHN_MUL_DAY-TEM-12001-%d%02d.TXT" % (yr, mon),
                           sep='\s+', header=None, names=['ID', 'Lat', 'Lon', 'Alt', 'Year', 'Mon', 'Day',
                                                          'Mean', 'Max', 'Min', 'QA_Mean', 'QA_Max', 'QA_Min'])

        if yr == 2017 and mon == 8:
            df.loc[df['ID'] == 58246, 'Lon'] = 12000

        if yr == 2018:
            df.loc[df['ID'] == 54287, 'Lon'] = 12800
            df.loc[df['ID'] == 54287, 'Lat'] = 4200

        df['Lat'] = df['Lat'].map(lambda x: int(str(x)[:-2]) + int(str(x)[-2:]) / 60)
        df['Lon'] = df['Lon'].map(lambda x: int(str(x)[:-2]) + int(str(x)[-2:]) / 60)

        df.loc[df['Alt'] > 1e5, 'Alt'] -= 1e5
        df['Alt'] *= .1

        df.loc[df['Mean'] == 32766, 'Mean'] = np.nan
        df.loc[df['Max'] == 32766, 'Max'] = np.nan
        df.loc[df['Min'] == 32766, 'Min'] = np.nan

        df['Mean'] *= .1
        df['Max'] *= .1
        df['Min'] *= .1

        unique_ls = df['ID'].unique()

        for i in trange(len(unique_ls), desc=r'Extract Tmp (%d-%02d)' % (yr, mon)):

            i = unique_ls[i]

            df_slice = df.loc[df['ID'] == i, :].copy()

            df_slice.loc[df_slice['QA_Mean'] != 0, 'Mean'] = np.nan
            df_slice.loc[df_slice['QA_Min'] != 0, 'Min'] = np.nan
            df_slice.loc[df_slice['QA_Max'] != 0, 'Max'] = np.nan

            df_slice['Mean'].interpolate(inplace=True)
            df_slice['Min'].interpolate(inplace=True)
            df_slice['Max'].interpolate(inplace=True)

            df_slice.index = df_slice.apply(lambda x: pd.to_datetime('%d-%02d-%02d' % (x['Year'], x['Mon'], x['Day'])), axis=1)

            dst_pth = r'G:\CMA\Station-TS\tmp\%s.csv' % i
            df_slice.to_csv(dst_pth, mode='a', header=True if not os.path.exists(dst_pth) else False)


# prc ---
for yr in range(2000, 2018):

    # yr = 2018
    for mon in range(8, 13):

        df = pd.read_table(r"G:\CMA\全国气象站日值数据\04降水\SURF_CLI_CHN_MUL_DAY-PRE-13011-%d%02d.TXT" % (yr, mon),
                           sep='\s+', header=None, names=['ID', 'Lat', 'Lon', 'Alt', 'Year', 'Mon', 'Day',
                                                          'Min', 'Max', 'Mean', 'QA_Min', 'QA_Max', 'QA_Mean'])

        if yr == 2017 and mon == 8:
            df.loc[df['ID'] == 58246, 'Lon'] = 12000

        if yr == 2018:
            df.loc[df['ID'] == 54287, 'Lon'] = 12800
            df.loc[df['ID'] == 54287, 'Lat'] = 4200

        df['Lat'] = df['Lat'].map(lambda x: int(str(x)[:-2]) + int(str(x)[-2:]) / 60)
        df['Lon'] = df['Lon'].map(lambda x: int(str(x)[:-2]) + int(str(x)[-2:]) / 60)

        df.loc[df['Alt'] > 1e5, 'Alt'] -= 1e5
        df['Alt'] *= .1

        df.loc[df['Mean'] == 32766, 'Mean'] = np.nan
        df.loc[df['Min'] == 32766, 'Min'] = np.nan
        df.loc[df['Max'] == 32766, 'Max'] = np.nan

        df.loc[df['Mean'] == 32700, 'Mean'] = 0
        df.loc[df['Max'] == 32700, 'Max'] = 0
        df.loc[df['Min'] == 32700, 'Min'] = 0

        df.loc[df['Mean'] >= 32000, 'Mean'] -= 32000
        df.loc[df['Mean'] >= 31000, 'Mean'] -= 31000
        df.loc[df['Mean'] >= 30000, 'Mean'] -= 30000

        df.loc[df['Min'] >= 32000, 'Min'] -= 32000
        df.loc[df['Min'] >= 31000, 'Min'] -= 31000
        df.loc[df['Min'] >= 30000, 'Min'] -= 30000

        df.loc[df['Max'] >= 32000, 'Max'] -= 32000
        df.loc[df['Max'] >= 31000, 'Max'] -= 31000
        df.loc[df['Max'] >= 30000, 'Max'] -= 30000
        
        df['Mean'] *= .1
        df['Max'] *= .1
        df['Min'] *= .1

        unique_ls = df['ID'].unique()

        for i in trange(len(unique_ls), desc=r'Extract Prc (%d-%02d)' % (yr, mon)):

            i = unique_ls[i]

            df_slice = df.loc[df['ID'] == i, :].copy()

            df_slice.loc[df_slice['QA_Mean'] != 0, 'Mean'] = np.nan
            df_slice.loc[df_slice['QA_Min'] != 0, 'Min'] = np.nan
            df_slice.loc[df_slice['QA_Max'] != 0, 'Max'] = np.nan

            # df_slice['Mean'].interpolate(inplace=True)
            # df_slice['Min'].interpolate(inplace=True)
            # df_slice['Max'].interpolate(inplace=True)

            df_slice['Mean'].fillna(0, inplace=True)
            # df_slice['Min'].fillna(0, inplace=True)
            # df_slice['Max'].fillna(0, inplace=True)

            df_slice.index = df_slice.apply(lambda x: pd.to_datetime('%d-%02d-%02d' % (x['Year'], x['Mon'], x['Day'])), axis=1)

            dst_pth = r'G:\CMA\Station-TS\prc\%s.csv' % i
            df_slice.to_csv(dst_pth, mode='a', header=True if not os.path.exists(dst_pth) else False)


# rhm ---
for yr in range(2000, 2018):

    # yr = 2018
    for mon in range(8, 13):

        df = pd.read_table(r"G:\CMA\全国气象站日值数据\08相对湿度\SURF_CLI_CHN_MUL_DAY-RHU-13003-%d%02d.TXT" % (yr, mon),
                           sep='\s+', header=None, names=['ID', 'Lat', 'Lon', 'Alt', 'Year', 'Mon', 'Day',
                                                          'Mean', 'Min', 'QA_Mean', 'QA_Min'])

        if yr == 2017 and mon == 8:
            df.loc[df['ID'] == 58246, 'Lon'] = 12000

        if yr == 2018:
            df.loc[df['ID'] == 54287, 'Lon'] = 12800
            df.loc[df['ID'] == 54287, 'Lat'] = 4200

        df['Lat'] = df['Lat'].map(lambda x: int(str(x)[:-2]) + int(str(x)[-2:]) / 60)
        df['Lon'] = df['Lon'].map(lambda x: int(str(x)[:-2]) + int(str(x)[-2:]) / 60)

        df.loc[df['Alt'] > 1e5, 'Alt'] -= 1e5
        df['Alt'] *= .1

        df.loc[df['Mean'] == 32766, 'Mean'] = np.nan
        df.loc[df['Min'] == 32766, 'Min'] = np.nan

        unique_ls = df['ID'].unique()

        for i in trange(len(unique_ls), desc=r'Extract Rhm (%d-%02d)' % (yr, mon)):

            i = unique_ls[i]

            df_slice = df.loc[df['ID'] == i, :].copy()

            df_slice.loc[df_slice['QA_Mean'] != 0, 'Mean'] = np.nan
            df_slice.loc[df_slice['QA_Mean'] != 0, 'Min'] = np.nan

            df_slice['Mean'].interpolate(inplace=True)
            # df_slice['Min'].interpolate(inplace=True)

            df_slice.index = df_slice.apply(lambda x: pd.to_datetime('%d-%02d-%02d' % (x['Year'], x['Mon'], x['Day'])), axis=1)

            dst_pth = r'G:\CMA\Station-TS\rhm\%s.csv' % i
            df_slice.to_csv(dst_pth, mode='a', header=True if not os.path.exists(dst_pth) else False)


# wind ---
for yr in range(2000, 2018):

    # yr = 2018
    for mon in range(8, 13):

        df = pd.read_table(r"G:\CMA\全国气象站日值数据\03风向风速\SURF_CLI_CHN_MUL_DAY-WIN-11002-%d%02d.TXT" % (yr, mon),
                           sep='\s+', header=None, names=['ID', 'Lat', 'Lon', 'Alt', 'Year', 'Mon', 'Day', 'Mean', 'Max',
                                                          'Max_Drt', 'Peak', 'Peak_Drt', 'QA_Mean', 'QA_Max',
                                                          'QA_Max_Drt', 'QA_Peak', 'QA_Peak_Drt'])

        if yr == 2017 and mon == 8:
            df.loc[df['ID'] == 58246, 'Lon'] = 12000

        if yr == 2018:
            df.loc[df['ID'] == 54287, 'Lon'] = 12800
            df.loc[df['ID'] == 54287, 'Lat'] = 4200

        df['Lat'] = df['Lat'].map(lambda x: int(str(x)[:-2]) + int(str(x)[-2:]) / 60)
        df['Lon'] = df['Lon'].map(lambda x: int(str(x)[:-2]) + int(str(x)[-2:]) / 60)

        df.loc[df['Alt'] > 1e5, 'Alt'] -= 1e5
        df['Alt'] *= .1

        df.loc[df['Mean'] == 32766, 'Mean'] = np.nan
        df.loc[df['Mean'] > 1e3, 'Mean'] -= 1e3
        df['Mean'] *= .1

        unique_ls = df['ID'].unique()

        for i in trange(len(unique_ls), desc=r'Extract Wnd (%d-%02d)' % (yr, mon)):

            i = unique_ls[i]

            df_slice = df.loc[df['ID'] == i, :].copy()

            df_slice.loc[df_slice['QA_Mean'] != 0, 'Mean'] = np.nan
            df_slice.loc[df_slice['QA_Mean'] != 0, 'Max'] = np.nan
            df_slice.loc[df_slice['QA_Mean'] != 0, 'Max_Drt'] = np.nan
            df_slice.loc[df_slice['QA_Mean'] != 0, 'Peak'] = np.nan
            df_slice.loc[df_slice['QA_Mean'] != 0, 'Peak_Drt'] = np.nan

            df_slice['Mean'].interpolate(inplace=True)
            # df_slice['Min'].interpolate(inplace=True)

            df_slice.index = df_slice.apply(lambda x: pd.to_datetime('%d-%02d-%02d' % (x['Year'], x['Mon'], x['Day'])), axis=1)

            dst_pth = r'G:\CMA\Station-TS\wnd\%s.csv' % i
            df_slice.to_csv(dst_pth, mode='a', header=True if not os.path.exists(dst_pth) else False)
