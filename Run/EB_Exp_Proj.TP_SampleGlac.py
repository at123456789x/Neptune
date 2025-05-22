# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: EB_Exp_Proj.TP_SampleGlac.py
@Time: 2025/5/21 22:00
@Function: 
"""

import sys
sys.path.append(r'/home/ritascake/wangq/ProjPy/Proj_SampleTP/Neptune/Module')

import geopandas as gpd
import rasterio as rio
# import schwimmbad
from Transfer_Coords import gener_coords
from ELR_Parameters import elr_params
from Reprojection_UTM import re_projection_utm
from Terrain_Parameters import terrain_params
from Meteor_Scen_Obs import *
from Distinguish_Pr import *
from Calc_Albedo import *
from Calc_Snow_Depth import *
from Calc_Solar_Rad import calc_solar_rad
from Calc_Longwave_Rad import calc_lw_rad
from Calc_Turbulence_Flux import calc_tbl_flux
from Calc_Cloud_Rad import *
from Hourly_Temperature_Interp import hr_temp_interp
from Generate_Init_Field import gener_init_filed
from Generate_Time_Series import gener_ts
from Torch_Albedo import *
import multiprocessing as mp
import warnings

plt.rc('font', family='Arial', size=14)
warnings.filterwarnings('ignore')

# ======================================================================================================================
# =======================================            Initialize model            =======================================
# ======================================================================================================================
# set env ---
rgi_pth = r'/home/ritascake/wangq/Proj_PIF/Shapefile/Glacier_2000.shp'
dem_pth = r'E:/NASA_DEM/Ease_Asia.tif'
gcm_pth = r'H:/CMIP6'
src_pth = r'/home/ritascake/wangq/MBG_Input/RGI_Glc'
output = r'/home/ritascake/wangq/Proj_PIF/Workshop/RS'
rgi_info_pth = r'/home/ritascake/wangq/MBG_Input/RGI_Glc_Info_Lmax.csv'

# forcing_elev_pth = r'E:/ERA5/geo_1279l4_0.1x0.1.grib2_v4_unpack.nc'
forcing_elev_pth = r'/home/ritascake/wangq/TPMFD/elevation_tpmfd.tif'
elr_pth = r'/home/ritascake/wangq/ERA5/Annual/ELR_2000_2020.nc'
sl_dif_pth = r'/home/ritascake/wangq/ERA5/Production/month_diffuse_ratio.nc'

# read the sampled mean ostrem curve ---
ostrem = pd.read_csv(r'/home/ritascake/wangq/MBG_Input/Ostrem.csv')

# Run Func ============================================================================================================
def EB_Main(cur_glc):

    # ==================================================================================================================
    # ========================================            Configuration            =====================================
    # ==================================================================================================================

    # glc info ---
    rgi = cur_glc.name
    cenlon, cenlat = cur_glc['TP_x'], cur_glc['TP_y']

    # config ---
    gcm, ssp = cur_glc['gcm'], cur_glc['ssp']
    spinup_yr = cur_glc['spinup.yrs']

    # meteorology ---
    deltaT_tag = cur_glc['deltaT.scale']

    if deltaT_tag == 'y':
        y_deltaT = cur_glc['deltaT.yr']
    elif deltaT_tag =='m':
        m_deltaT = cur_glc['deltaT.mon']

    deltaP = cur_glc['deltaP']
    deltaW = cur_glc['deltaW']
    deltaRH = cur_glc['deltaRH']

    dP = cur_glc['dP']
    dW = cur_glc['dW']
    dT = cur_glc['dT']
    dRH = cur_glc['dRH']

    # dst ---
    dst_pth = f'{os.path.dirname(output)}/Database/{rgi}'

    if not os.path.exists(dst_pth):
        os.makedirs(dst_pth)

    # parameters ---
    bgn_time = pd.to_datetime(cur_glc['BgnDate'], format='%Y%m%d')

    init_snow_depth = cur_glc['init.snow_depth']
    snow_depth_thres = cur_glc['snow.depth.threshold']
    snowfall_occur = cur_glc['snowfall.occur']

    snow_tag = cur_glc['albedo.scheme']

    with xr.open_dataset(sl_dif_pth) as ds_dif:
        dif_ratio = ds_dif.sel(longitude=cenlon, latitude=cenlat, method='Nearest')['__xarray_dataarray_variable__'].data

    utm_zone = int(int(cur_glc['geometry'].centroid.x) / 6) + 31
    dst_crs = 'epsg:326%s' % utm_zone

    cur_glc['buffer'] = gpd.GeoSeries(cur_glc['geometry'], crs='epsg:4326').copy().to_crs(dst_crs).buffer(300).to_crs('epsg:4326')[0]

    # r & re-projection raster to UTM zone ---
    if not os.path.exists(dst_pth + r'/thk_utm.tif'):
        re_projection_utm(rgi, dem_pth, cur_glc, src_pth, dst_pth, dst_crs)

    with xr.open_dataset(dst_pth + r'/dem_utm.tif') as utm_dem:
        xs, ys = np.meshgrid(utm_dem['x'].data, utm_dem['y'].data)

    # re-rf ras ---
    with rio.open(dst_pth + r'/dem_utm.tif') as dem_ras:
        dem_arr = dem_ras.read(1)
        meta = dem_ras.meta.copy()
        dem_arr[dem_arr == meta['nodata']] = -9999

    xss, yss = gener_coords(xs, ys, dst_crs, dst_pth)

    with rio.open(dst_pth + r'/thk_utm.tif') as thk_ras:
        thk_arr = thk_ras.read(1)

    # with rio.open(dst_pth + r'/thk_utm.tif') as thk_ras:
    #     thk_arr = rmask.mask(thk_ras, gpd.GeoSeries(cur_glc['geometry'], crs='epsg:4326').to_crs(
    #         thk_ras.meta['crs'].to_string()), crop=False)[0][0]

    # if os.path.exists(dst_pth + r'/debris_utm.tif'):
    #     with rio.open(dst_pth + r'/debris_utm.tif') as debris_ras:
    #         debris_arr = debris_ras.read(1)
    # else:
    #     debris_arr = np.zeros_like(dem_arr)

    # debris_arr[debris_arr < 0] = 0
    debris_arr = np.zeros_like(dem_arr)
    debris_arr *= 100
    red_arr = np.interp(debris_arr, ostrem['pred.x'], ostrem['value'])

    # plt.imshow(red_arr)
    # plt.colorbar()
    # plt.show()

    # debris_arr = np.zeros_like(dem_arr)

    # rvt module ---
    slope, aspect, f_sv = terrain_params(dem_arr)

    # ==================================================================================================================
    # =========================================            Climatology            ======================================
    # ==================================================================================================================

    # parameter ---
    if dT != 0:
        med_z = cur_glc['Med.z']
        elr_mean = dT * 1e3
        delta_T0 = ((dem_arr - med_z) / 1000 * elr_mean).copy()
        delta_T0[dem_arr == -9999] = np.nan

    else:
        med_z, elr_mean, delta_T0 = elr_params(forcing_elev_pth, dem_arr, elr_pth, cenlon, cenlat)

    try:

        df_meteor_cmip_bc = pd.read_csv(f"/home/ritascake/wangq/Proj_PIF/EC-Earth3-Veg-2300/BC_EC-Earth3-Veg_{ssp.lower()}.csv", index_col=0)
        df_meteor_cmip_bc.index = pd.period_range(df_meteor_cmip_bc.index.min(), df_meteor_cmip_bc.index.max(), freq='D')

        if deltaP != 1:
            df_meteor_cmip_bc *= deltaP
        if deltaW != 1:
            df_meteor_cmip_bc *= deltaW
        if deltaRH != 1:
            df_meteor_cmip_bc *= deltaRH

        # ==============================================================================================================
        # ===================================          Radiation simulation            =================================
        # ==============================================================================================================

        # plt.imshow(thk_arr)
        # plt.colorbar()
        # plt.show()

        # y, x = 250, 100         # south
        # df_rad_budget = pd.DataFrame()

        ts_init = pd.period_range(f'{2000-spinup_yr}-01-01', '2300-12-31', freq='M')
        ts_mb_arr = np.full((ts_init.size, dem_arr.shape[0], dem_arr.shape[1]), np.nan)

        if deltaT_tag == 'y':
            delta_T = delta_T0.copy() + y_deltaT

        wet_bulb, pr_solid = 0, 0

        # snow_depth_arr = np.zeros_like(dem_arr)
        snow_depth_arr = np.full_like(dem_arr, init_snow_depth * 10)
        snow_days_arr = np.full_like(dem_arr, 100)
        eb_abl_dt_arr = np.zeros_like(dem_arr)

        for init_i in trange(len(ts_init), desc='Loop Energy Balance of %s under %s (%s)' % (rgi, gcm, ssp)):

            init = ts_init[init_i]

            # break

            if deltaT_tag == 'm':
                delta_T = (delta_T0 + m_deltaT[init.month - 1]).copy()

            # ts_ser = pd.date_range('%s-%02d-01' % (init.year, init.month), periods=init.daysinmonth, freq='D')
            ts_ser = pd.period_range('%s-%02d-01' % (init.year, init.month), periods=init.daysinmonth, freq='D')

            eb_mb_arr = np.zeros((ts_ser.size, dem_arr.shape[0], dem_arr.shape[1]))

            for ts in range(ts_ser.size):

                dt = ts_ser[ts]

                # if dt == ts_ser[25]:
                #     break

                # Albedo ===========================================================================================
                shift_d = 7 - 1

                wet_bulb, pr_solid, tas_mean_dt, pr_total_dt, rh_mean_dt = (
                    calc_pr_type(wet_bulb, pr_solid, init_i, ts, dt, ts_ser, df_meteor_cmip_bc, dem_arr, delta_T, shift_d, dP, med_z))

                # print('Albedo Prediction.')

                albedo_arr = calc_albedo(df_meteor_cmip_bc, dt, delta_T, wet_bulb, shift_d, pr_solid, dem_arr, debris_arr, thk_arr, xss, yss)

                if snow_tag == 'DL':
                    albedo_arr[pr_solid.sum(axis=0) > 0.1] += .2
                    albedo_arr[albedo_arr > 1] = 1

                else:
                    # snow albedo ------
                    snow_depth_arr, snow_days_arr = calc_snow_depth(snow_depth_arr, pr_solid[-1], eb_abl_dt_arr, snow_days_arr, thk_arr, snowfall_occur)
                    snow_albedo_arr = calc_snow_albedo(snow_depth_arr, snow_days_arr, thk_arr, snow_depth_thres)

                    if snow_tag == 'AGE':
                        albedo_arr = snow_albedo_arr.copy()

                    elif snow_tag == 'COMB':
                        albedo_arr[snow_depth_arr > 0] = snow_albedo_arr[snow_depth_arr > 0]

                # print(np.nanmean(snow_depth_arr), np.nanmean(snow_days_arr), np.nanmean(snow_albedo_arr))

                # Temperature interp ===============================================================================
                tas_arr, tss_arr, tas_med = hr_temp_interp(df_meteor_cmip_bc, dt, delta_T)

                # Solar Rad ========================================================================================
                solar_rad_down_tot = calc_solar_rad(dem_arr, dt, cenlat, dif_ratio, slope, aspect, f_sv, albedo_arr, dst_pth)

                # Longwave Rad =====================================================================================
                LR_net, longwave_rad_down_day, longwave_rad_up_day = calc_lw_rad(tas_arr, tss_arr, f_sv)

                # Turbulence Flux ==================================================================================
                SH, LH = calc_tbl_flux(dem_arr, elr_mean, tas_arr, tss_arr, dt, df_meteor_cmip_bc, dW, med_z, tas_med[:, np.newaxis, np.newaxis])

                # Pr Heat ==========================================================================================
                PH = 4180 * (pr_total_dt / 24) * (tas_arr - tss_arr) / 3600

                # Cloud Rad ========================================================================================
                Cloud_net, cloud_fraction, cloud_albedo = calc_cloud_rad_wrf(rh_mean_dt[0], longwave_rad_down_day, dem_arr, elr_mean, tas_arr, med_z, tas_mean_dt[0])

                # Rad Budget =======================================================================================
                solar_rad_down_tot *= (1 - cloud_albedo)

                # normal ---
                SR_net = solar_rad_down_tot * (1 - albedo_arr[np.newaxis, :, :])            # * (1 - cloud_albedo)

                # Mass Balance =====================================================================================
                Rad_budget_net = SR_net + LR_net + Cloud_net - LH + SH + PH
                Rad_budget_net[(Rad_budget_net < 0) | (tas_arr < 273.15)] = 0

                # Refreeze =========================================================================================
                threshold = 274.65          # refreeze threshold
                k = 5 / (threshold - 273.15)
                b = -threshold * k

                Rad_budget_net[(tas_arr < threshold) & (tas_arr > 273.15)] = \
                    (Rad_budget_net[(tas_arr < threshold) & (tas_arr > 273.15)] *
                     (np.e ** (k * tas_arr[(tas_arr < threshold) & (tas_arr > 273.15)] + b)))

                Rad_budget_net = -Rad_budget_net.sum(axis=0) * 3600
                # Rad_budget_net[snow_depth_arr > 0] *= red_arr[snow_depth_arr > 0]         # debris reduction

                Sublimation = LH.copy()
                Sublimation[(Sublimation < 0) | (tas_arr > 273.15)] = 0
                Sublimation = Sublimation.sum(axis=0) * 3600 / 2.5e6

                eb_mb_arr[ts] = (Rad_budget_net / 3.34e5 + pr_solid[-1] + Sublimation).copy()

                eb_abl_dt_arr = eb_mb_arr[ts].copy()
                eb_abl_dt_arr[eb_abl_dt_arr > 0] = 0

                # print(np.nanmedian(eb_mb_arr[ts]))

                '''
                df_rad_budget.loc[dt, 'DR'] = solar_rad_down_tot[:, y, x].mean()
                df_rad_budget.loc[dt, 'UR'] = solar_rad_down_tot[:, y, x].mean() - SR_net[:, y, x].mean()
                df_rad_budget.loc[dt, 'SR.Net'] = SR_net[:, y, x].mean()
                df_rad_budget.loc[dt, 'Albedo'] = albedo_arr[y, x]
                df_rad_budget.loc[dt, 'Cloud.Fraction'] = cloud_fraction[:, y, x].mean()
                df_rad_budget.loc[dt, 'Cloud.Albedo'] = cloud_albedo[:, y, x].mean()
                df_rad_budget.loc[dt, 'CR'] = Cloud_net[:, y, x].mean()
                df_rad_budget.loc[dt, 'DLR'] = longwave_rad_down_day[:, y, x].mean() + df_rad_budget.loc[dt, 'CR']
                df_rad_budget.loc[dt, 'ULR'] = longwave_rad_up_day[:, y, x].mean()
                df_rad_budget.loc[dt, 'LR.Net'] = LR_net[:, y, x].mean() + df_rad_budget.loc[dt, 'CR']

                df_rad_budget.loc[dt, 'SH'] = SH[:, y, x].mean()
                df_rad_budget.loc[dt, 'LH'] = LH[:, y, x].mean()
                df_rad_budget.loc[dt, 'PH'] = PH[:, y, x].mean()
                df_rad_budget.loc[dt, 'Pr.sld'] = pr_solid[-1, y, x]
                df_rad_budget.loc[dt, 'Pr.sld3'] = pr_solid[-3:, y, x].sum()
                df_rad_budget.loc[dt, 'Pr'] = pr_total_dt[0]
                df_rad_budget.loc[dt, 'Tas'] = tas_arr[:, y, x].mean() - 273.15

                df_rad_budget.loc[dt, 'Net'] = (SR_net + LR_net + Cloud_net - LH + SH + PH)[:, y, x].mean()
                # '''

            # mon mb ---
            ts_mb_arr[init_i] = np.mean(eb_mb_arr, axis=0)

            if init.is_leap_year and init.month == 2:
                ts_mb_arr[init_i] *= (29 / 28)


        # export nc ====================================================================================================
        # pism spatial ---
        ice_loss_ser = np.where(np.isnan(ts_mb_arr), -1, ts_mb_arr / 86400)  # kg m-2 s-1

        df_meteor_cmip_bc_mon = df_meteor_cmip_bc.resample('M').mean().loc[
                                '%s-%02d' % (ts_init[0].year, ts_init[0].month):'%s-%02d' % (
                                    ts_init[-1].year, ts_init[-1].month)]

        tas_mean_ser = (((df_meteor_cmip_bc_mon['tmax'] + df_meteor_cmip_bc_mon['tmin']) * .5).
                        values[:, np.newaxis, np.newaxis] + delta_T[np.newaxis, :, :])

        # tas_mean_ser[tas_mean_ser > 273.15] = 273.15
        tas_mean_ser[np.isnan(tas_mean_ser)] = 250

        spatial_ds = gener_ts(ice_loss_ser, tas_mean_ser, ts_init, xs, ys, gcm, ssp, dst_pth)

        # init field ---
        if not os.path.exists(dst_pth + r'/Init_Field.nc'):
            gener_init_filed(xs, ys, xss, yss, thk_arr, ts_init, dem_arr, dst_pth)

    except Exception as e:
        print(str(e))


# Run ==================================================================================================================
if __name__ == '__main__':

    # Scenarios ---
    gcm = 'EC-Earth3-Veg'
    ssps = ['SSP245', 'SSP585']
    ssp = ssps[0]

    # rf ---
    rgi_shp = gpd.read_file(rgi_pth).set_index('RGIId')
    df_rgi = pd.read_csv(rgi_info_pth, index_col=0)

    # rgis = ['RGI60-13.54211']
    rgis = ['RGI60-13.49301']
    alias = ['PIF']

    threshes = [0]

    cur_id = 0
    rgi, thresh = rgis[cur_id], threshes[cur_id]
    # gcm = alias[cur_id]

    cur_glc = rgi_shp.loc[rgi, :]
    cur_glc['daylight_threshold'] = thresh

    cur_glc['gcm'] = gcm
    cur_glc['ssp'] = ssp

    cur_glc['TP_x'] = df_rgi.loc[rgi, 'TP_x']
    cur_glc['TP_y'] = df_rgi.loc[rgi, 'TP_y']

    cur_glc['2000s'] = df_rgi.loc[rgi, '2000s']
    cur_glc['2010s'] = df_rgi.loc[rgi, '2010s']

    # MLR ---
    cur_glc['dP'] = 0.053
    cur_glc['dT'] = -0.0064
    cur_glc['dRH'] = 0
    cur_glc['dW'] = 0.0017
    cur_glc['Med.z'] = 4700

    # delta ---
    cur_glc['deltaP'] = 1
    cur_glc['deltaW'] = 1
    cur_glc['deltaRH'] = 1

    # cur_glc['deltaT'] = 0
    # cur_glc['deltaT.mon'] = [0] * 12

    cur_glc['deltaT.mon'] = [-0.01, -0.64, 0.15, 0.4, -0.34, 0.4, 1.54, 2, 0.8, 0.1, -0.65, -1.84]
    cur_glc['deltaT.scale'] = 'm'      # y(ear) or m(onth)

    cur_glc['spinup.yrs'] = 5
    cur_glc['albedo.scheme'] = 'COMB'  # deep learning, age curve or combined (DL, AGE, COMB)

    cur_glc['init.snow_depth'] = 20  # cm
    cur_glc['snowfall.occur'] = 0.1  # mm
    cur_glc['snow.depth.threshold'] = 20  # mm

    # LOOP -------------------------------------------------------------------------------------------------------------
    rgi_pt_ls = []

    rgi_shp = rgi_shp.loc[rgi_shp['Area'] < 10]

    for i in range(len(rgi_shp)):

        for ssp in ssps:

            rgi = rgi_shp.index[i]
            alias = 'PIF'
            thresh = 0

            cur_glc = rgi_shp.loc[rgi, :]
            cur_glc['daylight_threshold'] = thresh

            cur_glc['gcm'] = gcm
            cur_glc['ssp'] = ssp

            cur_glc['TP_x'] = df_rgi.loc[rgi, 'TP_x']
            cur_glc['TP_y'] = df_rgi.loc[rgi, 'TP_y']

            cur_glc['2000s'] = df_rgi.loc[rgi, '2000s']
            cur_glc['2010s'] = df_rgi.loc[rgi, '2010s']

            # MLR ---
            cur_glc['dP'] = 0.053
            cur_glc['dT'] = -0.0064
            cur_glc['dRH'] = 0
            cur_glc['dW'] = 0.0017
            cur_glc['Med.z'] = 4700

            # delta ---
            cur_glc['deltaP'] = 1
            cur_glc['deltaW'] = 1
            cur_glc['deltaRH'] = 1

            # cur_glc['deltaT.yr'] = 0
            # cur_glc['deltaT.mon'] = [0] * 12

            cur_glc['deltaT.mon'] = [-0.01, -0.64, 0.15, 0.4, -0.34, 0.4, 1.54, 2, 0.8, 0.1, -0.65, -1.84]
            cur_glc['deltaT.method'] = 'm'  # y(ear) or m(onth)

            cur_glc['spinup.yrs'] = 5
            cur_glc['albedo.scheme'] = 'DL'  # deep learning, age curve or combined (DL, AGE, COMB)

            cur_glc['init.snow_depth'] = 20  # cm
            cur_glc['snowfall.occur'] = 0.1  # mm
            cur_glc['snow.depth.threshold'] = 20  # mm

            rgi_pt_ls.append(cur_glc)

    # Process Parallel ---
    Pool = mp.Pool()
    Pool.map(EB_Main, rgi_pt_ls)
