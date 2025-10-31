#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:03:35 2025

@author: lcarlton
"""
#%%
import pyvista as pv
import os
import cedalion
import cedalion.nirs
import xarray as xr
from cedalion import units
import gzip
import pickle
import numpy as np 
import cedalion.dataclasses as cdc 
from cedalion.io.forward_model import load_Adot

import matplotlib.pyplot as plt
pv.set_jupyter_backend('static')
# import my own functions from a different directory
import sys
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import spatial_basis_funs as sbf
import image_recon_func as irf

#%% set up config parameters
rec_str = 'conc_o'
noise_model = 'ar_irls'
fname_flag = 'mag'
C_meas_flag = True
t_win = [5, 7]
file_save = False
trial_type_img = ['right', 'left']
t_win = (5,8)
spatial_smoothing = True
sigma_smoothing = 50

if spatial_smoothing:
    smoothing_name = f'_smoothing-{sigma_smoothing}'
else:
    smoothing_name = ''

ROOT_DIR = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids/"
DATA_DIR = os.path.join(ROOT_DIR, 'derivatives', 'processed_data', 'image_space')
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'plots', 'image_space')

os.makedirs(SAVE_DIR, exist_ok=True)

PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw', 'ICBM152')
head_model = 'ICBM152'


#%% load head model 
head, PARCEL_DIR = irf.load_head_model('ICBM152', with_parcels=True)
Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))

#%% build plots 
threshold = -2 # log10 absolute
wl_idx = 1
M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)
SAVE = False
flag_hbo_list = [True] #, False] #, False]
flag_brain_list = [True] #, False]
flag_img_list = ['tstat','mag'] #, 'noise'] #['mag', 'tstat', 'noise'] #, 'noise'
flag_condition_list = ['right'] #, 'left']
scale = 1
flag = 'ts' # or mag

surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
surf = pv.wrap(surf.mesh)
      
cfg_list = [            
            {'alpha_meas': 1e4,
            'alpha_spatial': 1e-3,
            'DIRECT': False,
            'SB': False,
            'sigma_brain': 1,
            'sigma_scalp': 5},
            
            {'alpha_meas': 1e2,
            'alpha_spatial': 1e-3,
            'DIRECT': True,
            'SB': False,
            'sigma_brain': 1,
            'sigma_scalp': 5},
            
            {'alpha_meas': 1e4,
              'alpha_spatial': 1e-2,
              'DIRECT': False,
              'SB': True,
              'sigma_brain': 1,
              'sigma_scalp': 5},
            
            {'alpha_meas': 1e2,
            'alpha_spatial': 1e-2,
            'DIRECT': True,
            'SB': True,
            'sigma_brain': 1,
            'sigma_scalp': 5},
     ]

# flag_brain = True

for cfg in cfg_list[:1]:
    
    all_trial_X_hrf_mag = None
    # pdb.set_trace()
    F = None
    D = None
    G = None
    
    DIRECT = cfg['DIRECT']
    SB = cfg['SB']
    sigma_brain = cfg['sigma_brain']
    sigma_scalp = cfg['sigma_scalp']
    alpha_meas = cfg['alpha_meas']
    alpha_spatial = cfg['alpha_spatial']
    
    if DIRECT:
        direct_name = 'direct'
    else:
        direct_name = 'indirect'
    
    if C_meas_flag:
        Cmeas_name = 'Cmeas'
    else:
        Cmeas_name = 'noCmeas'
        
    if SB:
        filepath = os.path.join(DATA_DIR, f'image_hrf_{flag}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_Cmeas_{noise_model}{smoothing_name}.pkl.gz')
    else:
        filepath = os.path.join(DATA_DIR, f'image_hrf_{flag}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_Cmeas_{noise_model}{smoothing_name}.pkl.gz')
   

    with gzip.open( filepath, 'rb') as f:
          results = pickle.load(f)
    
    
    all_trial_X_tstat = results['X_tstat']
    all_trial_X_hrf_mag_weighted = results['X_hrf_ts_weighted']
    all_trial_X_stderr = results['X_std_err']
    if flag == 'ts':
        all_trial_X_tstat = all_trial_X_tstat.sel(time=slice(t_win[0], t_win[1])).mean('time')
        all_trial_X_hrf_mag_weighted = all_trial_X_hrf_mag_weighted.sel(time=slice(t_win[0], t_win[1])).mean('time')
        all_trial_X_stderr = all_trial_X_stderr.sel(time=slice(t_win[0], t_win[1])).mean('time')
    
    for flag_condition in flag_condition_list:
        
        for flag_img in flag_img_list:
            
            for flag_brain in flag_brain_list:
                
                for f, flag_hbo in enumerate(flag_hbo_list):
                    
                    p = pv.Plotter(shape=(1,1), window_size = [1000, 1000], off_screen=SAVE)
    
                    if flag_img == 'tstat':
                        foo_img = all_trial_X_tstat.sel(trial_type=flag_condition).copy()
                        # foo_img = foo_img.pint.dequantify()
                        # foo_img = foo_img.where( abs(foo_img)>t_crit, np.nan ) 
                        title_str = 't-stat'
                    elif flag_img == 'mag':
                        
                        foo_img = all_trial_X_hrf_mag_weighted.sel(trial_type=flag_condition).copy()
                        title_str = 'magnitude'
                    elif flag_img == 'noise':
                        foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
                        title_str = 'noise'
                        
                    foo_img = foo_img.pint.dequantify()
                    foo_img = foo_img.transpose('chromo', 'vertex')
                    
                    if flag_brain:
                        title_str = title_str + ' brain'
                        surface = 'brain'
                        foo_img = foo_img[:, Adot.is_brain.values]
                        foo_img[:, ~M[M.is_brain.values]] = np.nan
                        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
                        surf = pv.wrap(surf.mesh)
                              
                    else:
                        title_str = title_str + ' scalp'
                        surface = 'scalp'
                        foo_img = foo_img[:, ~Adot.is_brain.values]
                        foo_img[:, ~M[~M.is_brain.values]] = np.nan
                        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
                        surf = pv.wrap(surf.mesh)
                        
                    masked = foo_img.sel(chromo='HbO')
                    masked = masked.where(np.isfinite(masked))
                    clim = (-masked.max(skipna=True).values*scale, masked.max(skipna=True).values*scale)
    
                    if flag_hbo:
                        title_str = flag_condition + ': HbO'
                        foo_img = foo_img.sel(chromo='HbO')
                        chromo='HbO'
                    else:
                        title_str = flag_condition + ': HbR'
                        foo_img = foo_img.sel(chromo='HbR')
                        chromo='HbR'
                    
                    p.subplot(0, 0)
                    p.add_mesh(surf, scalars=foo_img, cmap=plt.cm.jet, clim=clim, show_scalar_bar=True, 
                               smooth_shading=True, nan_color=(0.9, 0.9, 0.9) )
                    p.camera_position = 'xy'
                    p.add_text(title_str, position='lower_left', font_size=10)
                    
                    if SAVE:
                        if SB:
                            img_folder =f'images_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{Cmeas_name}'
                        else:
                            img_folder = f'images_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_{Cmeas_name}'
                   
                        save_dir_tmp= os.path.join(SAVE_DIR, img_folder)
                        if not os.path.exists(save_dir_tmp):
                            os.makedirs(save_dir_tmp)
                            
                        file_name = f'IMG_{flag}_{flag_condition}_{flag_img}_{chromo}_{surface}_scale-{scale}{smoothing_name}.png'
                        p.screenshot( os.path.join(save_dir_tmp, file_name) )
                        p.close()
                    else:
                        p.show()
    
    

# %%
