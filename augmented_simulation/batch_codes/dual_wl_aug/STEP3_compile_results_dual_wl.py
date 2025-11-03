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
TASK = 'RS'
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

FWHM_HbO_direct = xr.DataArray(np.zeros([len(alpha_meas_list), len(alpha_spatial_list), len(sigma_brain_list), len(sigma_scalp_list), len(VERTEX_LIST)]),
                      dims = ['alpha_meas', 'alpha_spatial', 'sigma_brain', 'sigma_scalp', 'vertex'],
                      coords = {'alpha_meas': alpha_meas_list,
                                'alpha_spatial': alpha_spatial_list,
                                'sigma_brain': sigma_brain_list,
                                'sigma_scalp': sigma_scalp_list,
                                'vertex': VERTEX_LIST,
                                 })

CNR_HbO_direct = FWHM_HbO_direct.copy()
CNR_HbO_indirect = FWHM_HbO_direct.copy()
FWHM_HbO_indirect = FWHM_HbO_direct.copy()
crosstalk_brainVscalp_direct = FWHM_HbO_direct.copy()
LE_HbO_direct = FWHM_HbO_direct.copy()
crosstalk_brainVscalp_indirect = FWHM_HbO_direct.copy()
LE_HbO_indirect = FWHM_HbO_direct.copy()
contrast_ratio_HbO_direct = FWHM_HbO_direct.copy()
contrast_ratio_HbO_indirect = FWHM_HbO_direct.copy()
crosstalk_HbOVHbR_direct = FWHM_HbO_direct.copy()
crosstalk_HbRVHbO_direct = FWHM_HbO_direct.copy()
crosstalk_HbOVHbR_indirect = FWHM_HbO_direct.copy()
crosstalk_HbRVHbO_indirect = FWHM_HbO_direct.copy()

for sigma_brain in sigma_brain_list: 
    
    for sigma_scalp in sigma_scalp_list:
        
        if sigma_brain == 0 and sigma_scalp != 0:
            continue
        elif sigma_brain != 0 and sigma_scalp == 0:
            continue
        
        for alpha_meas in alpha_meas_list:
            
            for alpha_spatial in alpha_spatial_list:
                
                with open(os.path.join(BATCH_DIR, f'COMPILED_METRIC_RESULTS_{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_sb-{float(sigma_brain)}_ss-{float(sigma_scalp)}_am-{float(alpha_meas)}_as-{float(alpha_spatial)}_{GLM_METHOD}_dual_wl.pkl'), 'rb') as f:
                    RESULTS = pickle.load(f)
                
                FWHM_HbO_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['FWHM_HbO_direct'].squeeze()
                FWHM_HbO_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['FWHM_HbO_indirect'].squeeze()
                CNR_HbO_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['CNR_HbO_direct'].squeeze()
                CNR_HbO_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['CNR_HbO_indirect'].squeeze()
                crosstalk_brainVscalp_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['crosstalk_brainVscalp_HbO_direct'].squeeze()
                crosstalk_brainVscalp_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['crosstalk_brainVscalp_HbO_indirect'].squeeze()
                LE_HbO_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['localization_error_HbO_direct'].squeeze()
                LE_HbO_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['localization_error_HbO_indirect'].squeeze()
                crosstalk_HbOVHbR_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['crosstalk_HbOVHbR_direct'].squeeze()
                crosstalk_HbRVHbO_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['crosstalk_HbRVHbO_direct'].squeeze()
                crosstalk_HbOVHbR_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['crosstalk_HbOVHbR_indirect'].squeeze()
                crosstalk_HbRVHbO_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['crosstalk_HbRVHbO_indirect'].squeeze()
                contrast_ratio_HbO_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['contrast_ratio_HbO_indirect'].squeeze()
                contrast_ratio_HbO_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, :] = RESULTS['contrast_ratio_HbO_direct'].squeeze()


RESULTS = {
           'crosstalk_HbOVHbR_direct': crosstalk_HbOVHbR_direct,
           'crosstalk_HbRVHbO_direct':  crosstalk_HbRVHbO_direct,
           'crosstalk_HbOVHbR_indirect': crosstalk_HbOVHbR_indirect,
           'crosstalk_HbRVHbO_indirect': crosstalk_HbRVHbO_indirect,
           'CNR_HbO_direct': CNR_HbO_direct,
           'CNR_HbO_indirect': CNR_HbO_indirect,
           'FWHM_HbO_indirect': FWHM_HbO_indirect,
           'FWHM_HbO_direct': FWHM_HbO_direct,
           'localization_error_HbO_indirect': LE_HbO_indirect,
           'localization_error_HbO_direct': LE_HbO_direct,
           'crosstalk_brainVscalp_HbO_direct': crosstalk_brainVscalp_direct,
           'crosstalk_brainVscalp_HbO_indirect': crosstalk_brainVscalp_indirect,
            'contrast_ratio_HbO_direct': contrast_ratio_HbO_direct,
            'contrast_ratio_HbO_indirect': contrast_ratio_HbO_indirect,
           }
  
                                    
with open(os.path.join(SAVE_DIR, f'COMPILED_METRIC_RESULTS_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_{GLM_METHOD}_dual_wl.pkl'), 'wb') as f:
     pickle.dump(RESULTS, f)



# %%
