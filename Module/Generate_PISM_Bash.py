# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: W.Chiung@foxmail.com
@Software: PyCharm
@File: Generate_PISM_Bash.py
@Time: 2024/1/7 2:50
@Function: 
"""
import pandas as pd
import xarray as xr

def gener_bash(cpu_core, pism_nc, ts, init_field_pth):

    with xr.open_dataset(pism_nc, decode_times=False) as ds:
        nc_dims = dict(ds.dims)
        bgn_t = ds['time'].attrs['units'].split(' ')[-1]
        end_t = str(ts[-1] + pd.to_timedelta(1, 'D')).split(' ')[0]

    rgi_id = pism_nc.split('/')[-3]
    gcm = pism_nc.split('/')[-1].split('_')[0]
    ssp = pism_nc.split('/')[-1].split('_')[-1].split('.')[0]
    workshop_prefix = '/'.join(pism_nc.split('/')[:-1])

    bash_pth = f'{workshop_prefix}/bash-pism-{gcm}-{ssp}.sh'
    ts_pth = f'{workshop_prefix}/PISM-Ts-{gcm}_{ssp}_{ts[-1].year + 1}.nc'
    ex_pth = f'{workshop_prefix}/PISM-Ex-{gcm}_{ssp}_{ts[-1].year + 1}.nc'
    # fn_pth = f'{workshop_prefix}/PISM-Fn-{gcm}_{ssp}_{ts[-1].year + 1}.nc'
    fn_pth = f'{workshop_prefix}/PISM-Fn-{gcm}_{ssp}.nc'

    with open(bash_pth, 'w', newline='\n') as f:

        # f.write('source /datanew/hejp/wangq-env/env/bashrc.sh')

        f.write('ncap2 -O -s \'time@bounds="time_bnds"; defdim("nv",2); '
                'time_bnds=array(0,0,/$time,$nv/); time_bnds(:,0)=time; '
                'time_bnds(:-2,1)=time(1:); time_bnds(-1,1)=2*time(-1)-time(-2);\' '
                '%s '
                '%s\n' % (f'{workshop_prefix}/{gcm}_{ssp}.nc',
                          f'{workshop_prefix}/{gcm}_{ssp}-bounds.nc'))

        f.write('mpirun -np %s pismr -regional -i %s -bootstrap '
                '-ys %s -ye %s '
                '-Mx %s -My %s '
                '-Mz 2001 -Mbz 1 -Lz 2000 -z_spacing equal -skip '
                '-grid.recompute_longitude_and_latitude false '
                '-grid.registration corner -surface given '
                '-surface_given_file %s '
                '-stress_balance ssa+sia -stress_balance.ice_free_thickness_standard 1 '
                '-bed_smoother_range 0 -no_model_strip 0 -pseudo_plastic -dry '
                '-ts_file %s '
                '-ts_times monthly '
                '-extra_file %s '
                '-extra_times monthly '
                '-extra_vars ice_surface_temp,mask,thk,topg,usurf,velsurf_mag,velbase_mag,'
                'climatic_mass_balance,lon,lat '
                '-o %s '
                '-view thk,velsurf_mag'
                '\n' % (cpu_core,
                        init_field_pth,
                        bgn_t,
                        end_t,
                        nc_dims['x'],
                        nc_dims['y'],
                        f'{workshop_prefix}/{gcm}_{ssp}-bounds.nc',
                        f'{ts_pth}',
                        f'{ex_pth}',
                        f'{fn_pth}'))

        return bash_pth, ts_pth, ex_pth, fn_pth
