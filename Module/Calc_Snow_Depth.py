# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Calc_Snow_Depth.py
@Time: 2024/7/18 20:58
@Function: 
"""

import numpy as np

def calc_snow_depth(snow_depth_arr, snowfall_arr, ablation_dt_arr, snow_days_arr, thk_arr, snowfall_occur_thres):

    snow_depth_arr = snow_depth_arr + snowfall_arr
    snow_depth_arr += ablation_dt_arr
    snow_depth_arr[snow_depth_arr < 0] = 0

    snow_depth_arr[thk_arr == 0] = np.nan

    snow_days_arr += 1
    snow_days_arr[snowfall_arr >= snowfall_occur_thres] = 0
    # snow_days_arr[snowfall_arr >= 5] = 0
    # snow_days_arr[snowfall_arr >= .1] = 0

    return snow_depth_arr, snow_days_arr
