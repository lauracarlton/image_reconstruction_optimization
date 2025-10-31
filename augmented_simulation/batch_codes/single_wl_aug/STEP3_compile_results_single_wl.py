#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile all the results from the batch jobs 
- ensure the directories match where the data was saved 
- adjust the lists of image recon parameters to match what was used in the simulations

configure the following parameters:
    - ROOT_DIR: should point to a BIDs data folder
    - HEAD_MODEL: which atlas to use - options in cedalion are Colin27 and ICBM152
    - GLM_METHOD: which solving method was used in preprocessing of augmented data - ols or ar_irls
    - TASK: which of the tasks in the BIDS dataset was augmented 
    - BLOB_SIGMA: the standard deviation of the Gaussian blob of activation (mm) * MUST HAVE UNITS *
    - SCALE_FACTOR: the amplitude of the maximum change in 850nm OD in channel space
    - VERTEX_LIST: list of seed vertices to be used 
    - exclude_subj: any subjects IDs within the BIDs dataset to be excluded
    
choose the image recon parameters to test 
- alpha_meas_list: select range of alpha measurement
- alpha_spatial_list: select range of alpha spatial 
- sigma_brain_list: select range of sigma brain 
- sigma_scalp_list: select range of sigma scalp 

@author: lcarlton
"""
#%%
import pickle
import os
import xarray as xr
import numpy as np

ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids')
BLOB_SIGMA = 15
GLM_METHOD = 'ols'
VERTEX_LIST = [10089, 10453, 14673, 11323, 13685, 11702, 8337]
SCALE_FACTOR = 0.02
alpha_meas_list = [10 ** i for i in range(-1, 3)]
alpha_spatial_list = [1e-3, 1e-2]
sigma_brain_list = [0, 1, 3, 5]
sigma_scalp_list = [0, 1, 5, 10, 20]
exclude_subj = ['sub-577']


SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
BATCH_DIR = os.path.join(SAVE_DIR, 'batch_reuslts')

dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if 'sub' in d and d not in exclude_subj]

FWHM = xr.DataArray(np.zeros([len(alpha_meas_list), len(alpha_spatial_list), len(sigma_brain_list), len(sigma_scalp_list), len(VERTEX_LIST)]),
                      dims = ['alpha_meas', 'alpha_spatial', 'sigma_brain', 'sigma_scalp', 'vertex'],
                      coords = {'alpha_meas': alpha_meas_list,
                                'alpha_spatial': alpha_spatial_list,
                                'sigma_brain': sigma_brain_list,
                                'sigma_scalp': sigma_scalp_list,
                                'vertex': VERTEX_LIST,
                                 })

CNR = FWHM.copy()
crosstalk_brainVscalp = FWHM.copy()
localization_error = FWHM.copy()
perc_recon_brain = FWHM.copy()
perc_recon_scalp = FWHM.copy()
contrast_ratio = FWHM.copy()

for sigma_brain in sigma_brain_list:
    
    for sigma_scalp in sigma_scalp_list:
        
        if sigma_brain == 0 and sigma_scalp != 0:
            continue
        elif sigma_brain != 0 and sigma_scalp == 0:
            continue
        
        for alpha_meas in alpha_meas_list:
            
            for alpha_spatial in alpha_spatial_list:
                
                with open(os.path.join(BATCH_DIR, f'COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_sb-{float(sigma_brain)}_ss-{float(sigma_scalp)}_am-{float(alpha_meas)}_as-{float(alpha_spatial)}_{GLM_METHOD}_single_wl.pkl'), 'rb') as f:
                    RESULTS = pickle.load(f)
                
                FWHM.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['FWHM'].squeeze()
                CNR.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['CNR'].squeeze()
                crosstalk_brainVscalp.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['crosstalk_brainVscalp'].squeeze()
                localization_error.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['localization_error'].squeeze()
                perc_recon_brain.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['perc_recon_brain'].squeeze()
                perc_recon_scalp.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['perc_recon_scalp'].squeeze()
                contrast_ratio.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['contrast_ratio'].squeeze()


RESULTS = {
           'CNR': CNR,
           'FWHM': FWHM,
           'localization_error': localization_error,
           'crosstalk_brainVscalp': crosstalk_brainVscalp,
           'perc_recon_brain': perc_recon_brain,
           'perc_recon_scalp': perc_recon_scalp,
            'contrast_ratio': contrast_ratio,
           }
  
                                    
with open(os.path.join(SAVE_DIR, f'COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_{GLM_METHOD}_single_wl.pkl'), 'wb') as f:
     pickle.dump(RESULTS, f)



# %%
