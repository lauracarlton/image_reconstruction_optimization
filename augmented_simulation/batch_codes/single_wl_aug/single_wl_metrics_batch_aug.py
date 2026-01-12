#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_wl_metrics_batch_aug.py

Batch processing script for single-wavelength augmented simulation metrics. This
module is designed to be called by a batch job scheduler (e.g., SGE, SLURM) to
compute image reconstruction metrics for a single parameter combination. It
generates synthetic single-wavelength fNIRS measurements, performs image
reconstruction, and computes quality metrics.

Usage
-----
This script is typically called by STEP2_submit_single_wl_aug_batch_job.py via a
batch scheduler with command-line arguments::

    python single_wl_metrics_batch_aug.py <alpha_meas> <alpha_spatial> <sigma_brain> <sigma_scalp>

Command-line Arguments
----------------------
- alpha_meas (float): Measurement regularization parameter value.
- alpha_spatial (float): Spatial regularization parameter value.
- sigma_brain (float): Brain spatial basis function width in mm.
- sigma_scalp (float): Scalp spatial basis function width in mm.

Configurables (defaults shown)
-----------------------------
Data Storage Parameters:
- ROOT_DIR (str): '/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids_v2'
    - Root directory containing forward model and variance data.
- EXCLUDED (list[str]): ['sub-577']
    - Subject IDs to skip during processing.

Head Model Parameters:
- HEAD_MODEL (str): 'ICBM152'
    - Head model used for forward modeling (options: 'Colin27', 'ICBM152').

GLM Parameters:
- NOISE_MODEL (str): 'ar_irls'
    - GLM method used in variance estimation step (options: 'ols', 'ar_irls').
- TASK (str): 'RS'
    - Task identifier matching the variance estimation dataset.

HRF Parameters:
- BLOB_SIGMA (pint Quantity): 15 * units.mm
    - Standard deviation of the Gaussian used for spatial activation blob.
- SCALE_FACTOR (float): 0.02
    - Amplitude of synthetic activation in optical density units.
- VERTEX_LIST (list[int]): [10089, 10453, 14673, 11323, 13685, 11702, 8337]
    - List of seed vertex indices for synthetic activation generation.

Fixed Parameters:
- WL_IDX (int): 1
    - Wavelength index for single-wavelength simulation (0=760nm, 1=850nm).
- lambda_R (float): 1e-6
    - scaling parameter for the image prior used in reconstruction.
- mask_threshold (float): -2
    - Log of sensitivity threshold for creating brain/scalp mask.

Outputs
-------
- Individual pickle file saved to <ROOT_DIR>/derivatives/cedalion/augmented_data/batch_results/single_wl/
  with filename: COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_sb-{sigma_brain}_ss-{sigma_scalp}_am-{alpha_meas}_as-{alpha_spatial}_lR-{lambda_R}_{NOISE_MODEL}_single_wl.pkl
  containing dictionary with metrics:
  - FWHM: Full width at half maximum of reconstructed activation
  - CNR: Contrast-to-noise ratio
  - crosstalk_brainVscalp: Brain vs scalp crosstalk measure
  - localization_error: Distance between true and reconstructed peak
  - perc_recon_brain: Percentage of signal reconstructed in brain
  - perc_recon_scalp: Percentage of signal reconstructed in scalp
  - contrast_ratio: Ratio of reconstructed to expected contrast

Dependencies
------------
- cedalion, xarray, numpy, scipy, pickle, sys, modules/image_recon_func,
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

from cedalion import nirs, io, units, xrutils
import cedalion.sim.synthetic_hrf as synthetic_hrf
from cedalion.io.forward_model import load_Adot

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import spatial_basis_func as sbf
import image_recon_func as irf
import get_image_metrics as gim  

warnings.filterwarnings('ignore')

#%% SETUP CONFIGS
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids_v2')
HEAD_MODEL = 'ICBM152'
NOISE_MODEL = 'ar_irls'
TASK = 'RS'
BLOB_SIGMA = 15 * units.mm
SCALE_FACTOR = 0.02
WL_IDX = 1
VERTEX_LIST = [10089, 10453, 14673, 11323, 13685, 11702, 8337]
EXCLUDED = ['sub-577']
lambda_R = 0.25e-6

#%% SETUP DOWNSTREAM CONFIGS
CMEAS_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
SAVE_DIR = os.path.join(CMEAS_DIR, 'batch_results', 'single_wl')
PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw', 'probe')

os.makedirs(SAVE_DIR, exist_ok=True)

mask_threshold = -2

dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if 'sub' in d and d not in EXCLUDED]

#%% GET INPUTS
alpha_meas = float(sys.argv[1])
alpha_spatial = float(sys.argv[2])
sigma_brain = float(sys.argv[3])
sigma_scalp = float(sys.argv[4])
# alpha_meas = 1e5
# alpha_spatial = 0.01
# sigma_brain = 1.0
# sigma_scalp = 5.0

print(f"Processing alpha_meas - {alpha_meas}, alpha_spatial - {alpha_spatial}, sigma_brain - {sigma_brain}, sigma_scalp - {sigma_scalp}") 
    
#%% LOAD DATA
subj_temp =  f'{subject_list[0]}/nirs/{subject_list[0]}_task-{TASK}_run-01_nirs.snirf'
file_name = os.path.join(ROOT_DIR, subj_temp)
rec = io.read_snirf(file_name)[0]

head, parcel_dir = irf.load_head_model(with_parcels=False)
Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))

channels = Adot.channel
n_chs = len(channels)
A_fw = Adot.isel(wavelength = 1)
nV_brain = A_fw.is_brain.sum().values

#%% LOAD IN CMEAS
with open(os.path.join(CMEAS_DIR, f"C_meas_subj_task-{TASK}_blob-{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_{NOISE_MODEL}.pkl"), 'rb') as f:
    C_meas_list = pickle.load(f)
    
#%% GET THE METRICS FOR ALL VERTICES

FWHM = xr.DataArray(np.zeros( len(VERTEX_LIST)),
                      dims = ['vertex'],
                      coords = {
                                'vertex': VERTEX_LIST,
                                 })

CNR = FWHM.copy()
crosstalk_brainVscalp = FWHM.copy()
localization_error = FWHM.copy()
perc_predicted_brain = FWHM.copy()
perc_predicted_scalp = FWHM.copy()
contrast_ratio = FWHM.copy()

M = sbf.get_sensitivity_mask(Adot, mask_threshold, 1)

if sigma_brain > 0 and sigma_scalp > 0:
    print(f'\tsigma brain = {sigma_brain}, sigma scalp = {sigma_scalp}')
    SB = True
    G_brain_path = os.path.join(PROBE_DIR, f'G_matrix_sigmabrain-{sigma_brain}.pkl')
    if os.path.exists(G_brain_path):
        with open(G_brain_path, 'rb') as f:
            G_brain = pickle.load(f)
    else:
        brain_downsampled = sbf.downsample_mesh(head.brain.vertices, 
                                                M[M.is_brain], 
                                                sigma_brain*units.mm)

        G_brain = sbf.get_kernel_matrix(brain_downsampled, 
                                        head.brain.vertices, 
                                        sigma_brain*units.mm)

        with open(G_brain_path, 'wb') as f:
                pickle.dump(G_brain, f)

    G_scalp_path = os.path.join(PROBE_DIR, f'G_matrix_sigmascalp-{sigma_scalp}.pkl')
    if os.path.exists(G_scalp_path):
        with open(G_scalp_path, 'rb') as f:
            G_scalp = pickle.load(f)
    else:
        scalp_downsampled = sbf.downsample_mesh(head.scalp.vertices, 
                                                M[~M.is_brain], 
                                                sigma_scalp*units.mm)

        G_scalp = sbf.get_kernel_matrix(scalp_downsampled, 
                                        head.scalp.vertices, 
                                        sigma_scalp*units.mm)

        with open(G_scalp_path, 'wb') as f:
                pickle.dump(G_scalp, f)

    G = {'G_brain': G_brain,
         'G_scalp': G_scalp}
    
    H_single_wl = sbf.get_H(G, Adot)
    A_single_wl = H_single_wl.copy()
    nkernels_brain = G_brain.kernel.shape[0]

else:
    SB=False
    G = None
    A_single_wl = Adot.copy()
 
F_indirect = None
D_indirect = None
max_eig_indirect=None
            
for vv, seed_vertex in enumerate(VERTEX_LIST):
    
    print(f'\tseed vertex = {vv+1}/{len(VERTEX_LIST)}')

    all_subj_X_hrf_mag = None

    for subject in subject_list:
        
        print(f'\t\t\tsubject: {subject}')
        
        C_meas = C_meas_list.sel(vertex=seed_vertex, subject=subject)

        W_indirect, D_indirect, F_indirect, max_eig_indirect = irf.calculate_W(A_single_wl, 
                                                                                lambda_R=lambda_R, 
                                                                                alpha_meas=alpha_meas, 
                                                                                alpha_spatial=alpha_spatial, 
                                                                                DIRECT=False, 
                                                                                C_meas_flag=True, 
                                                                                C_meas=C_meas, 
                                                                                D=D_indirect, 
                                                                                F=F_indirect, 
                                                                                max_eig=max_eig_indirect)
       
        W_single_wl = W_indirect.isel(wavelength=WL_IDX)

        blob_img = synthetic_hrf.build_blob_from_seed_vertex(head, vertex = seed_vertex, scale = BLOB_SIGMA)
        y =  A_fw[:, A_fw.is_brain.values] @ blob_img 

        b = SCALE_FACTOR / y.max()
        y = y * b 
        expected_contrast = blob_img * b
        
        # get image
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
        

        cov_img_diag = irf._get_image_noise_post_indirect(A_single_wl, 
                                                            W_indirect, 
                                                            alpha_spatial=alpha_spatial, 
                                                            lambda_R=lambda_R,
                                                            SB=SB, 
                                                            G=G)
        
        X_mse = cov_img_diag.isel(wavelength=WL_IDX)
        
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
            
    # get the average
    X_hrf_mag_mean_weighted = X_hrf_mag_weighted / X_mse_inv_weighted
    X_hrf_mag_mean = all_subj_X_hrf_mag.mean('subj')
    X_mse_mean_within_subject = 1 / X_mse_inv_weighted
                        
    X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_mag - X_hrf_mag_mean_weighted)**2 / all_subj_X_mse # X_mse_subj_tmp is weights for each sub
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp.mean('subj')
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects * X_mse_mean_within_subject # / (all_subj_X_mse**-1).mean('subj')
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
    FWHM.loc[seed_vertex] = gim.get_FWHM(X_brain, head, version='weighted_mean')
     
    ##### GET CNR ####
    CNR.loc[seed_vertex] = X_brain[ROI].mean() / X_err_brain[ROI].mean()
    contrast_ratio.loc[seed_vertex] = X_brain.max() / expected_contrast.max()
    
    ##### GET LOCALIZATION ERROR ####   
    origin = head.brain.vertices[seed_vertex,:]
    localization_error.loc[seed_vertex] = gim.get_localization_error(origin, X_brain, head)
    
    ##### GET CROSSTALK ####   
    crosstalk_brainVscalp.loc[seed_vertex], _ = gim.get_crosstalk(X_brain, X_scalp)
    
    ##### GET PERCENT RECONSTRUCTED ####
    perc_predicted_brain.loc[seed_vertex],  perc_predicted_scalp.loc[seed_vertex] = gim.get_percent_reconstructed(X_brain.values, X_scalp.values, y, A_fw)
    
RESULTS = {'FWHM': FWHM,
           'CNR': CNR,
           'crosstalk_brainVscalp': crosstalk_brainVscalp,
           'localization_error': localization_error,
           'perc_recon_brain': perc_predicted_brain,
           'perc_recon_scalp': perc_predicted_scalp,
           'contrast_ratio': contrast_ratio
    }

with open(os.path.join(SAVE_DIR, f'COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_sb-{sigma_brain}_ss-{sigma_scalp}_am-{alpha_meas}_as-{alpha_spatial}_lR-{lambda_R}_{NOISE_MODEL}_single_wl.pkl'), 'wb') as f:
    pickle.dump(RESULTS, f)
 
print('Job Complete.')
# %%
