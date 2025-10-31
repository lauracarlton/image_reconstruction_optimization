#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 11:53:41 2025

@author: lcarlton
"""
#%%
import pickle
import os 
import gzip 
import sys 

import xarray as xr
import pyvista as pv 
import numpy as np

import matplotlib.pyplot as plt
import cedalion.dataclasses as cdc 

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import image_recon_func as irf
import get_image_metrics as gim  

plt.rcParams['font.size'] = 50


#%% CONFIG
ROOT_DIR = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids/"
DATA_DIR = os.path.join(ROOT_DIR, 'derivatives', 'processed_data', 'image_space')
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'plots', 'image_space')
PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw', 'ICBM152')

C_meas_flag = True
fname_flag = 'ts'
SAVE = True
PLOT_mask = True
ROI_selection = 'tstat'

rec_str = 'conc_o'
glm_method = 'ar_irls'

head, PARCEL_DIR = irf.load_head_model('ICBM152', with_parcels=False)
Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))

cfg_list = [{'alpha_meas': 1e4,
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
            'sigma_scalp': 5}
     ]


#%% GENERATE THE PLOT
if PLOT_mask:
    p0_left = pv.Plotter(shape=(2,4), window_size = [2000, 700], off_screen=SAVE)
    p0_right = pv.Plotter(shape=(2,4), window_size = [2000, 700], off_screen=SAVE)

for cc, cfg in enumerate(cfg_list):
    
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
        
    
    # import timeseries
    if SB:
        filepath = os.path.join(DATA_DIR, f'image_hrf_ts_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{Cmeas_name}_{glm_method}.pkl.gz')
    else:
        filepath = os.path.join(DATA_DIR, f'image_hrf_ts_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_{Cmeas_name}_{glm_method}.pkl.gz')
    
    
    with gzip.open(filepath, 'rb') as f:
        ts_results = pickle.load(f)
         
    # import image magnitude 
    if SB:
        filepath = os.path.join(DATA_DIR, f'image_hrf_mag_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{Cmeas_name}_{glm_method}.pkl.gz')
    else:
        filepath = os.path.join(DATA_DIR, f'image_hrf_mag_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_{Cmeas_name}_{glm_method}.pkl.gz')
    
    
    with gzip.open(filepath, 'rb') as f:
        image_results = pickle.load(f)
         
    
    if SB:
        title = f'{direct_name},\nsigma_brain-{sigma_brain}, sigma_scalp-{sigma_scalp},\nalpha_meas-{alpha_meas}, alpha_spatial-{alpha_spatial}'
    else:
        title = f'{direct_name},\nalpha_meas-{alpha_meas}, alpha_spatial-{alpha_spatial}'

    #% plot the timeseries based on an ROI selection
    mag = image_results['X_hrf_ts_weighted']
    tstat = image_results['X_tstat']
    is_brain = Adot.is_brain
    
    
    for trial_type in mag.trial_type:

        # select top vertices based on the magnitude or tstat image 
        mag_hbo = mag.sel(trial_type=trial_type, chromo='HbO') #.isel(vertex=mag.is_brain.values)
        tstat_hbo = tstat.sel(trial_type=trial_type, chromo='HbO').drop_vars('chromo').pint.dequantify()
       
        if ROI_selection == 'mag':
            # OPTION A: get the max vertices 
            ROI_brain = gim.get_ROI(mag_hbo.sel(vertex=is_brain), 0.5)
            ROI_scalp = gim.get_ROI(mag_hbo.sel(vertex=~is_brain), 0.5)
        elif ROI_selection == 'tstat':
            ROI_brain = gim.get_ROI(tstat_hbo.sel(vertex=is_brain), 0.8)
            ROI_scalp = gim.get_ROI(tstat_hbo.sel(vertex=~is_brain), 0.8)            
            
        # 1. plot the ROI mask
        if PLOT_mask:
            if trial_type=='right':
                p0 = p0_right
            else:
                p0 = p0_left
                
            mask = xr.zeros_like(mag_hbo.sel(vertex=is_brain))
            
            mask[ROI_brain] = 1
            surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
            surf = pv.wrap(surf.mesh)   
            p0.subplot(0, cc)
            p0.add_mesh(surf, scalars=mask, cmap=plt.cm.jet, clim=[-1,1], show_scalar_bar=False, 
                        nan_color=(0.9,0.9,0.9), smooth_shading=True)
            p0.add_text(title, position='lower_left', font_size=10)
            p0.camera_position = 'xy'
            
            mask = xr.zeros_like(mag_hbo.sel(vertex=~is_brain))
            
            mask[ROI_scalp] = 1
            surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
            surf = pv.wrap(surf.mesh)   
            p0.subplot(1, cc)
            p0.add_mesh(surf, scalars=mask, cmap=plt.cm.jet, clim=[-1,1], show_scalar_bar=False, 
                        nan_color=(0.9,0.9,0.9), smooth_shading=True)
            p0.add_text(title, position='lower_left', font_size=10)
            p0.camera_position = 'xy'
            
        # get the weighted group average using each subject's timeseries
        
        # get the mean within the BRAIN ROI 
        X_hrf_ts = ts_results['X_hrf_ts'].sel(vertex=is_brain, trial_type=trial_type).pint.dequantify()
        X_mse_ts = ts_results['X_mse'].sel(vertex=is_brain, trial_type=trial_type).pint.dequantify()
      
        X_ts_roi = X_hrf_ts.sel(vertex=ROI_brain)  
        X_mse_ts_roi = X_mse_ts.sel(vertex=ROI_brain)  
        
        X_ts_roi_mean = (X_ts_roi/X_mse_ts_roi).sum('vertex') / (1/X_mse_ts_roi).sum('vertex')
        X_mse_roi_mean =  1 / (1/X_mse_ts_roi).sum('vertex')

        
        # get the subject mean 
        X_mean_within_subj =  1 / (1/X_mse_roi_mean).sum('subj')
        
        X_group_ts = (X_ts_roi_mean/X_mse_roi_mean).sum('subj') / (1/X_mse_roi_mean).sum('subj')    
        
        X_between_subj = (X_ts_roi_mean - X_group_ts)**2  
        X_between_subj = X_between_subj / X_mse_roi_mean  
        X_between_subj = X_between_subj.mean('subj') * X_mean_within_subj # normalized by the within subject variances as weights
      
        X_mse_tot_subj = X_mse_roi_mean + X_between_subj
        
        X_weighted_group_ts = (X_ts_roi_mean/X_mse_tot_subj).sum('subj') / (1/X_mse_tot_subj).sum('subj')
        
        mse_total = 1 / (1/X_mse_tot_subj).sum('subj')
        
        X_std_err = np.sqrt(mse_total) #/np.sqrt(len(X_mse_tot_subj.subj))
             
        X_ts_roi_brain = X_weighted_group_ts
        X_var_roi_brain = X_std_err
        
        # get the mean within the SCALP ROI 
        X_hrf_ts = ts_results['X_hrf_ts'].sel(vertex=~is_brain, trial_type=trial_type).pint.dequantify()
        X_mse_ts = ts_results['X_mse'].sel(vertex=~is_brain, trial_type=trial_type).pint.dequantify()
        
        X_ts_roi = X_hrf_ts.sel(vertex=ROI_scalp)  
        X_mse_ts_roi = X_mse_ts.sel(vertex=ROI_scalp)  
        
        # weighted by vertex variances per subject
        X_ts_roi_mean = (X_ts_roi/X_mse_ts_roi).sum('vertex') / (1/X_mse_ts_roi).sum('vertex')
        X_mse_roi_mean =  1 / (1/X_mse_ts_roi).sum('vertex')
        
        X_mean_within_subj =  1 / (1/X_mse_roi_mean).sum('subj')
        
        X_group_ts = (X_ts_roi_mean/X_mse_roi_mean).sum('subj') / (1/X_mse_roi_mean).sum('subj')    
        
        X_between_subj = (X_ts_roi_mean - X_group_ts)**2  
        X_between_subj = X_between_subj / X_mse_roi_mean  
        X_between_subj = X_between_subj.mean('subj') * X_mean_within_subj # normalized by the within subject variances as weights
      
        X_mse_tot_subj = X_mse_roi_mean + X_between_subj
        
        X_weighted_group_ts = (X_ts_roi_mean/X_mse_tot_subj).sum('subj') / (1/X_mse_tot_subj).sum('subj')
        
        mse_total = 1 / (1/X_mse_tot_subj).sum('subj')
        
        X_std_err = np.sqrt(mse_total) 
    
        X_ts_roi_scalp =  X_weighted_group_ts.pint.dequantify()
        X_var_roi_scalp = X_std_err
        
        max_tstat = X_ts_roi_brain.sel(chromo='HbO', time=slice(5,8)).mean('time') / X_var_roi_brain.sel(chromo='HbO', time=slice(5,8)).mean('time')

        fig, ax = plt.subplots(2,1,figsize=[20,20], sharey=True)
        ax[0].plot(X_ts_roi_brain.time, X_ts_roi_brain.sel(chromo='HbO'), 'r', lw=8, label = 'HbO')
        ax[0].plot(X_ts_roi_brain.time, X_ts_roi_brain.sel(chromo='HbR'), 'b', lw=8, label = 'HbR')
        
        ax[1].plot(X_ts_roi_scalp.time, X_ts_roi_scalp.sel(chromo='HbO'), 'r', lw=8, label = 'HbO')
        ax[1].plot(X_ts_roi_scalp.time, X_ts_roi_scalp.sel(chromo='HbR'), 'b', lw=8, label = 'HbR')
        
        ax[0].text(0.97, 0.95, fr'$t_{{\mathrm{{peak}}}} = {np.round(max_tstat.values, 2)}$',
                   transform=ax[0].transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                  
        
        ax[0].fill_between(X_ts_roi_brain.time, X_ts_roi_brain.sel(chromo='HbO') - X_var_roi_brain.sel(chromo='HbO'),
                        X_ts_roi_brain.sel(chromo='HbO') + X_var_roi_brain.sel(chromo='HbO'),
                        color='red', alpha=0.3)
        ax[0].fill_between(X_ts_roi_brain.time, X_ts_roi_brain.sel(chromo='HbR') - X_var_roi_brain.sel(chromo='HbR'), 
                        X_ts_roi_brain.sel(chromo='HbR') + X_var_roi_brain.sel(chromo='HbR'),
                        color='blue', alpha=0.3)
        
        ax[1].fill_between(X_ts_roi_scalp.time, X_ts_roi_scalp.sel(chromo='HbO') - X_var_roi_scalp.sel(chromo='HbO'),
                        X_ts_roi_scalp.sel(chromo='HbO') + X_var_roi_scalp.sel(chromo='HbO'),
                        color='red', alpha=0.3)
        ax[1].fill_between(X_ts_roi_scalp.time, X_ts_roi_scalp.sel(chromo='HbR') - X_var_roi_scalp.sel(chromo='HbR'), 
                        X_ts_roi_scalp.sel(chromo='HbR') + X_var_roi_scalp.sel(chromo='HbR'),
                        color='blue', alpha=0.3)
        
        ax[1].set_xlabel('Time (s)')
        ax[0].set_ylabel('Concentration (M)')
        ax[1].set_ylabel('Concentration (M)')
        ax[0].set_title('Brain')
        ax[1].set_title('Scalp')
        ax[1].grid('on')
        ax[0].grid('on')
        plt.tight_layout()
    
        if SB:
            filepath = os.path.join(SAVE_DIR, f'ts_plot_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{Cmeas_name}_{trial_type.values}_roi-{ROI_selection}_{glm_method}-legdrift.png')
        else:
            filepath = os.path.join(SAVE_DIR, f'ts_plot_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_{Cmeas_name}_{trial_type.values}_roi-{ROI_selection}_{glm_method}-legdrift.png')
            
        plt.savefig(filepath, dpi=300)

        
if PLOT_mask:
    
    if SAVE:
            
        filepath = os.path.join(SAVE_DIR, f'image_mask_{Cmeas_name}_right_roi-{ROI_selection}_{glm_method}.png')
        p0_right.screenshot(filepath)
        
        filepath = os.path.join(SAVE_DIR, f'image_mask_{Cmeas_name}_left_roi-{ROI_selection}_{glm_method}.png')
        p0_left.screenshot(filepath)
        
    else:
        p0_right.show()
# %%
