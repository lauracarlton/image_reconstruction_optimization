#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG5_STEP2_get_single_wavelength_image_metrics.py

Single-wavelength augmented simulation and image reconstruction metrics. This
module generates synthetic fNIRS measurements at a single wavelength with
known ground truth activations, performs image reconstruction with various
regularization parameters, and computes quantitative metrics to evaluate reconstruction quality.

Usage
-----
Edit the CONFIG section (ROOT_DIR, VERTEX_LIST, alpha_meas_list, etc.) then run::

    python FIG5_STEP2_get_single_wavelength_image_metrics.py

Inputs
------
- Forward model file (Adot.nc) containing sensitivity matrix.
- Measurement variance estimates (C_meas) from FIG5&6_STEP1.
- Seed vertex indices for generating synthetic activations.

Configurables (defaults shown)
-----------------------------
Data Storage Parameters:
- ROOT_DIR (str): 
    - Root BIDS directory containing forward model and variance data.
- exclude_subj (list[str]): ['sub-577']
    - Subject IDs to skip during processing.

Head Model Parameters:
- HEAD_MODEL (str): 'ICBM152'
    - Head model used for forward modeling.
- MASK_THRESHOLD (float): -2
    - Log of sensitivity threshold for creating brain/scalp mask.

HRF Parameters:
- VERTEX_LIST (list[int]): [10089, 10453, 14673, 11323, 13685, 11702, 8337]
    - List of seed vertex indices for synthetic activation generation.
- BLOB_SIGMA (pint Quantity): 15 * units.mm
    - Standard deviation of the Gaussian used for spatial activation blob.
- SCALE_FACTOR (float): 0.02
    - Amplitude of synthetic activation in optical density units.

Wavelength Parameters:
- WL_IDX (int): 1
    - Wavelength index for single-wavelength simulation (0 or 1, corresponding to 760nm or 850nm).

GLM Parameters:
- NOISE_MODEL (str): 'ols'
    - GLM method used in variance estimation step.
- TASK (str): 'RS'
    - Task identifier matching the variance estimation dataset.

Image Reconstruction Parameters to Test:
- alpha_meas_list (list[float]): [1e-4, 1, 1e4]
    - Range of measurement regularization parameters to sweep.
- alpha_spatial_list (list[float]): [1e-2, 1e-3]
    - Range of spatial regularization parameters to sweep.
- sigma_brain_list (list[pint Quantity]): [0, 5] * units.mm
    - Range of brain spatial basis function widths to test.
- sigma_scalp_list (list[pint Quantity]): [0, 10] * units.mm
    - Range of scalp spatial basis function widths to test.

Outputs
-------
- Gzipped pickle file saved to <ROOT_DIR>/derivatives/cedalion/augmented_data/ with filename:
  COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_{NOISE_MODEL}_single_wl.pkl
  containing xarray DataArrays with the following metrics indexed by 
  [alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, vertex]:
  - FWHM: Full width at half maximum of reconstructed activation
  - CNR: Contrast-to-noise ratio
  - crosstalk_brainVscalp: Brain vs scalp crosstalk measure
  - localization_error: Distance between true and reconstructed peak
  - perc_recon_brain: Percentage of signal reconstructed in brain
  - contrast_ratio: Ratio of reconstructed to expected contrast

Dependencies
------------
- cedalion, xarray, numpy, scipy, pickle, modules/image_recon_func,
  modules/get_image_metrics, modules/spatial_basis_funs

Author: Laura Carlton
"""

#---------------------------------------------------
#%% IMPORT MODULES
#---------------------------------------------------
import os
import sys
import pickle
import warnings

import numpy as np 
import xarray as xr

import cedalion.sim.synthetic_hrf as synthetic_hrf
from cedalion import units, io
from cedalion.io.forward_model import load_Adot

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import image_recon_func as irf
import spatial_basis_func as sbf
import get_image_metrics as gim  

warnings.filterwarnings('ignore')

#%% CONFIG 
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids')
HEAD_MODEL = 'ICBM152'
NOISE_MODEL = 'ols'
TASK = 'RS'
BLOB_SIGMA = 15 * units.mm
SCALE_FACTOR = 0.02
WL_IDX = 1 # use wavelength at index 1 (850 nm according to our dataset)
VERTEX_LIST = [10089, 10453, 14673, 11323, 13685, 11702, 8337]
EXCLUDED = ['sub-577']

# IMAGE RECON PARAMS TO TEST 
alpha_meas_list = [1e-4, 1, 1e4] 
alpha_spatial_list = [1e-2, 1e-3] 
sigma_brain_list = [0, 5] *units.mm 
sigma_scalp_list = [0, 10] *units.mm 

#%% SETUP DOWNSTREAM CONFIGS
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw', HEAD_MODEL)

dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if 'sub' in d and d not in EXCLUDED]

# HEAD PARAMS
MASK_THRESHOLD = -2 # log of sensitivity threshold 

#%% LOAD DATA
head, parcel_dir = irf.load_head_model(HEAD_MODEL, with_parcels=False)
Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))

channels = Adot.channel
n_chs = len(channels)
A_fw = Adot.isel(wavelength = 1)
nV_brain = A_fw.is_brain.sum().values

#%% LOAD IN CMEAS
with open(os.path.join(SAVE_DIR, f"C_meas_subj_task-{TASK}_blob-{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_{NOISE_MODEL}.pkl"), 'rb') as f:
    C_meas_list = pickle.load(f)
    
#%% GET THE METRICS FOR ALL VERTICES
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
perc_predicted_brain = FWHM.copy()
contrast_ratio = FWHM.copy()
    
M = sbf.get_sensitivity_mask(Adot, MASK_THRESHOLD, WL_IDX)

print('Running simulation using a single wavelength')
for sigma_brain in sigma_brain_list: 

    for sigma_scalp in sigma_scalp_list:
        
        if sigma_scalp > 0 and sigma_brain > 0:
            print(f'\tsigma brain = {sigma_brain.magnitude}, sigma scalp = {sigma_scalp.magnitude}')

            G_brain_path = os.path.join(PROBE_DIR, f'G_matrix_sigmabrain-{float(sigma_brain.magnitude)}.pkl')
            if os.path.exists(G_brain_path):
                with open(G_brain_path, 'rb') as f:
                    G_brain = pickle.load(f)
            else:
                brain_downsampled = sbf.downsample_mesh(head.brain.vertices, M[M.is_brain], sigma_brain)
                G_brain = sbf.get_kernel_matrix(brain_downsampled, head.brain.vertices, sigma_brain)
                with open(G_brain_path, 'wb') as f:
                        pickle.dump(G_brain, f)

            G_scalp_path = os.path.join(PROBE_DIR, f'G_matrix_sigmascalp-{float(sigma_scalp.magnitude)}.pkl')
            if os.path.exists(G_scalp_path):
                with open(G_scalp_path, 'rb') as f:
                    G_scalp = pickle.load(f)
            else:
                scalp_downsampled = sbf.downsample_mesh(head.scalp.vertices, M[~M.is_brain], sigma_scalp)
                G_scalp = sbf.get_kernel_matrix(scalp_downsampled, head.scalp.vertices, sigma_scalp)
                with open(G_scalp_path, 'wb') as f:
                        pickle.dump(G_scalp, f)

            G = {'G_brain': G_brain,
                 'G_scalp': G_scalp}
    
            H_single_wl = sbf.get_H(G, Adot)
            A_single_wl = H_single_wl.copy()
            nkernels_brain = G_brain.kernel.shape[0]
            
        elif sigma_scalp == 0 and sigma_brain ==0:
            G = None
            A_single_wl = Adot.copy()
            
        else:
            continue

        for alpha_spatial in alpha_spatial_list:
           
            print(f'\t\talpha_spatial = {alpha_spatial}')
            
            F_direct = None
            D_direct = None
            F_indirect = None
            D_indirect = None

            for ii, seed_vertex in enumerate(VERTEX_LIST):
                print(f'\t\t\tseed vertex = {ii+1}/{len(VERTEX_LIST)}')

                for alpha_meas in alpha_meas_list:
                    # alpha_meas = 1e5
                    print(f'\t\t\t\talpha_meas = {alpha_meas}')
                    all_subj_X_hrf_mag = None
                    
                    for subject in subject_list:
                        print(f'\t\t\t\t\tsubject: {subject}')
                        
                        #### SINGLE WAVELENGTH IMAGE
                        C_meas = C_meas_list.sel(vertex=seed_vertex, subject=subject)

                        # get the pseudoinverse
                        W_indirect, D_indirect, F_indirect = irf.calculate_W(A_single_wl, alpha_meas, alpha_spatial, DIRECT=False,
                                                     C_meas_flag=True, C_meas=C_meas, D=D_indirect, F=F_indirect)
                       
                        W_single_wl = W_indirect.isel(wavelength=WL_IDX)

                        # get the image of the blob
                        blob_img = synthetic_hrf.build_blob_from_seed_vertex(head, vertex = seed_vertex, scale = BLOB_SIGMA)
                        y =  A_fw[:, A_fw.is_brain.values] @ blob_img 
                        b = SCALE_FACTOR / y.max()
                        y = y * b 
                        expected_contrast = blob_img * b    

                        # get the noise free image
                        X = W_single_wl.values @ y.values
                        
                        if sigma_brain > 0:
                            sb_X_brain = X[:nkernels_brain]
                            sb_X_scalp = X[nkernels_brain:]
                            
                            # PROJECT BACK TO SURFACE SPACE 
                            X_brain = G_brain.values.T @ sb_X_brain
                            X_scalp = G_scalp.values.T @ sb_X_scalp
                        else:
                            X_brain = X[:nV_brain]
                            X_scalp = X[nV_brain:]
                        
                        # get noise 
                        X = xr.DataArray(np.hstack([X_brain, X_scalp]),
                                         dims=('vertex'),
                                         coords={'is_brain':('vertex', A_fw.is_brain.values)})
                        
                        cov_img_tmp = W_single_wl *np.sqrt(C_meas.isel(wavelength=WL_IDX).values) # W is pseudo inverse  --- diagonal (faster than W C W.T)
                        cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
                        
                        if sigma_brain > 0:
                            cov_img_diag = sbf.go_from_kernel_space_to_image_space_indirect(cov_img_diag, G)
           
                        X_mse = X.copy()
                        X_mse.values = cov_img_diag
                        
                        if all_subj_X_hrf_mag is None:
                            all_subj_X_hrf_mag = X
                            all_subj_X_hrf_mag = all_subj_X_hrf_mag.assign_coords(subj=subject)

                            all_subj_X_mse = X_mse
                            all_subj_X_mse = all_subj_X_mse.assign_coords(subj=subject)

                            X_hrf_mag_weighted = X / X_mse
                            X_mse_inv_weighted = 1 / X_mse
                            
                        else:
                            X_hrf_mag_tmp = X.assign_coords(subj=subject)
                            X_mse_tmp = X_mse.assign_coords(subj=subject)

                            all_subj_X_hrf_mag = xr.concat([all_subj_X_hrf_mag, X_hrf_mag_tmp], dim='subj')
                            all_subj_X_mse = xr.concat([all_subj_X_mse, X_mse_tmp], dim='subj')

                            X_hrf_mag_weighted = X_hrf_mag_weighted + X_hrf_mag_tmp / X_mse
                            X_mse_inv_weighted = X_mse_inv_weighted + 1 / X_mse  
                            
                        
                        # END OF SUBJECT LOOP

                    # get the average
                    X_hrf_mag_mean_weighted = X_hrf_mag_weighted / X_mse_inv_weighted
                    X_hrf_mag_mean = all_subj_X_hrf_mag.mean('subj')
                    X_mse_mean_within_subject = 1 / X_mse_inv_weighted
                                        
                    X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_mag - X_hrf_mag_mean_weighted)**2 / all_subj_X_mse # X_mse_subj_tmp is weights for each sub
                    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp.mean('subj')
                    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects * X_mse_mean_within_subject
                    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects.pint.dequantify()
                    
                    # get the weighted average
                    mse_btw_within_sum_subj = all_subj_X_mse + X_mse_weighted_between_subjects
                    denom = (1/mse_btw_within_sum_subj).sum('subj')
                    
                    X_hrf_mag_mean_weighted = (all_subj_X_hrf_mag / mse_btw_within_sum_subj).sum('subj')
                    X_hrf_mag_mean_weighted = X_hrf_mag_mean_weighted / denom
                    
                    mse_total = 1/denom
                    mse_total = mse_total.where(np.isfinite(mse_total), np.nan)
                    X_stderr_weighted = np.sqrt( mse_total )

                    ### GET SINGLE WAVELENGTH METRICS
                    X_brain = X_hrf_mag_mean_weighted[X_hrf_mag_mean_weighted.is_brain.values]
                    X_scalp = X_hrf_mag_mean_weighted[~X_hrf_mag_mean_weighted.is_brain.values]
                    X_err_brain = X_stderr_weighted[X_hrf_mag_mean_weighted.is_brain.values]
                    
                    ROI = gim.get_ROI(X_brain, 0.5)
                    ##### GET FWHM ####
                    FWHM.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = gim.get_FWHM(X_brain, head, version='weighted_mean')
                     
                    ##### GET CNR ####
                    CNR.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = X_brain[ROI].mean() / X_err_brain[ROI].mean()
                    contrast_ratio.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = X_brain.max() / expected_contrast.max()

                    ##### GET LOCALIZATION ERROR ####   
                    origin = head.brain.vertices[seed_vertex,:]
                    localization_error.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = gim.get_localization_error(origin, X_brain, head)
                    
                    ##### GET CROSSTALK ####   
                    crosstalk_brainVscalp.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex], _ = gim.get_crosstalk(X_brain, X_scalp)
                    
                    ##### GET PERCENT RECONSTRUCTED ####
                    perc_predicted_brain.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex], _ = gim.get_percent_reconstructed(X_brain.values, X_scalp.values, y, A_fw)
                    
RESULTS = {'FWHM': FWHM,
           'CNR': CNR,
           'crosstalk_brainVscalp': crosstalk_brainVscalp,
           'localization_error': localization_error,
           'perc_recon_brain': perc_predicted_brain,
           'contrast_ratio': contrast_ratio
    }

print('saving the data')
with open(os.path.join(SAVE_DIR, f'COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_{NOISE_MODEL}_single_wl.pkl'), 'wb') as f:
    pickle.dump(RESULTS, f)

# %%
