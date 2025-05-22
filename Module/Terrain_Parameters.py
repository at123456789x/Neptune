# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Terrain_Parameters.py
@Time: 2023/12/1 17:43
@Function: 
"""

import numpy as np
import rvt.vis
import rvt.default

# rvt module ---
def terrain_params(dem_arr):

    slope_aspect = rvt.vis.slope_aspect(dem_arr, 30, 30, 'radian')  # degree

    slope, aspect = slope_aspect['slope'], slope_aspect['aspect']

    aspect = -aspect
    aspect[aspect <= 0] = -aspect[aspect <= 0] - np.pi
    aspect[aspect >= 0] = np.pi - aspect[aspect >= 0]

    f_sv = rvt.vis.sky_view_factor(dem_arr, 30, compute_svf=True, svf_n_dir=16, svf_r_max=10, svf_noise=1)['svf']

    return slope, aspect, f_sv
