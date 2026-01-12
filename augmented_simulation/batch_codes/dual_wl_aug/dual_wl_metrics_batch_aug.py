#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dual_wl_metrics_batch_aug.py

Batch processing script for dual-wavelength augmented simulation metrics. This
module is designed to be called by a batch job scheduler (e.g., SGE, SLURM) to
compute image reconstruction metrics for a single parameter combination. It
generates synthetic dual-wavelength fNIRS measurements, performs image
reconstruction, and computes quality metrics.

Usage
-----
This script is typically called by STEP2_submit_dual_wl_aug_batch_job.py via a
batch scheduler with command-line arguments::

    python dual_wl_metrics_batch_aug.py <alpha_meas> <alpha_spatial> <sigma_brain> <sigma_scalp>

Command-line Arguments
----------------------
- alpha_meas (float): Measurement regularization parameter value.
- alpha_spatial (float): Spatial regularization parameter value.
- sigma_brain (float): Brain spatial basis function width in mm.
- sigma_scalp (float): Scalp spatial basis function width in mm.

Configurables (defaults shown)
-----------------------------
Data Storage Parameters:
- ROOT_DIR (str): 
    - Root directory containing forward model and variance data.
- EXCLUDED (list[str]): ['sub-577']
    - Subject IDs to skip during processing.

Head Model Parameters:
- HEAD_MODEL (str): 'ICBM152'
    - Head model used (options: 'Colin27', 'ICBM152').

GLM Parameters:
- NOISE_MODEL (str): 'ols'
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
- wl_idx (int): 1
    - Wavelength index used for mask generation.
- mask_threshold (float): -2
    - Log of sensitivity threshold for creating brain/scalp mask.
- chromo_list (list[str]): ['HbO', 'HbR']
    - Chromophores to reconstruct and evaluate.

Outputs
-------
- Individual pickle file saved to <ROOT_DIR>/derivatives/cedalion/augmented_data/batch_results/
  with filename: COMPILED_METRIC_RESULTS_{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_sb-{sigma_brain}_ss-{sigma_scalp}_am-{alpha_meas}_as-{alpha_spatial}_{NOISE_MODEL}_dual_wl.pkl
  containing dictionary with metrics:
  - crosstalk_HbOVHbR_direct/indirect: HbO-to-HbR crosstalk ratio
  - crosstalk_HbRVHbO_direct/indirect: HbR-to-HbO crosstalk ratio
  - CNR_HbO_direct/indirect: Contrast-to-noise ratio for HbO
  - contrast_ratio_HbO_direct/indirect: Ratio of reconstructed to expected contrast
  - FWHM_HbO_direct/indirect: Full width at half maximum of HbO activation
  - localization_error_HbO_direct/indirect: Distance between true and reconstructed peak
  - crosstalk_brainVscalp_HbO_direct/indirect: Brain vs scalp crosstalk

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
VERTEX_LIST = [10089, 10453, 14673, 11323, 13685, 11702, 8337]
EXCLUDED = ['sub-577']
lambda_R = 1e-6

#%% SETUP DOWNSTREAM CONFIGS
CMEAS_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
SAVE_DIR = os.path.join(CMEAS_DIR, 'batch_results', 'dual_wl')
PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw', 'probe')

os.makedirs(SAVE_DIR, exist_ok=True)

wl_idx = 1
mask_threshold = -2
chromo_list = ['HbO', 'HbR']

dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if 'sub' in d and d not in EXCLUDED]

#%% GET INPUTS
alpha_meas = float(sys.argv[1])
alpha_spatial = float(sys.argv[2])
sigma_brain = float(sys.argv[3])
sigma_scalp = float(sys.argv[4])
# alpha_meas = 1e2
# alpha_spatial = 0.001
# sigma_brain = 0.0
# sigma_scalp = 0.0

print(f"Processing alpha_meas - {alpha_meas}, alpha_spatial - {alpha_spatial}, sigma_brain - {sigma_brain}, sigma_scalp - {sigma_scalp}") 
    
#%% LOAD DATA
subj_temp =  f'{subject_list[0]}/nirs/{subject_list[0]}_task-{TASK}_run-01_nirs.snirf'
file_name = os.path.join(ROOT_DIR, subj_temp)
rec = io.read_snirf(file_name)[0]

head, parcel_dir = irf.load_head_model(with_parcels=False)
Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))

channels = Adot.channel

n_chs = len(channels)
Adot_stacked = irf.get_Adot_scaled(Adot, Adot.wavelength)

nV_brain = Adot.is_brain.sum().values
nV_scalp = (~Adot.is_brain).sum().values
nV = len(Adot.vertex)

ec = nirs.get_extinction_coefficients("prahl", rec['amp'].wavelength)
einv = xrutils.pinv(ec)

#%% LOAD IN CMEAS
with open(CMEAS_DIR + f"/C_meas_subj_task-{TASK}_blob-{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_{NOISE_MODEL}.pkl", 'rb') as f:
    C_meas_list = pickle.load(f)

#%% GET THE METRICS FOR ALL VERTICES

FWHM_HbO_direct = xr.DataArray(np.zeros( len(VERTEX_LIST)),
                      dims = ['vertex'],
                      coords = {
                                'vertex': VERTEX_LIST,
                                 })

CNR_HbO_direct = FWHM_HbO_direct.copy()
contrast_ratio_HbO_direct = FWHM_HbO_direct.copy()
crosstalk_brainVscalp_HbO_direct = FWHM_HbO_direct.copy()
localization_error_HbO_direct = FWHM_HbO_direct.copy()

FWHM_HbO_indirect = FWHM_HbO_direct.copy()
CNR_HbO_indirect = FWHM_HbO_direct.copy()
contrast_ratio_HbO_indirect = FWHM_HbO_direct.copy()
crosstalk_brainVscalp_HbO_indirect = FWHM_HbO_direct.copy()
localization_error_HbO_indirect = FWHM_HbO_direct.copy()

P_direct = xr.DataArray(np.zeros([len(chromo_list), len(VERTEX_LIST)]),
                      dims = ['chromo', 'vertex'],
                      coords = {
                                'chromo': chromo_list,
                                'vertex': VERTEX_LIST,
                                 })
   
C_direct = P_direct.copy()
C_indirect = P_direct.copy()
P_indirect = P_direct.copy()

M = sbf.get_sensitivity_mask(Adot, mask_threshold, wl_idx)

if sigma_brain > 0 and sigma_scalp > 0:
    print(f'\tsigma brain = {sigma_brain}, sigma scalp = {sigma_scalp}')

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

    SB=True
    H_single_wl = sbf.get_H(G, Adot)
    H_dual_wl = sbf.get_H_stacked(G, Adot_stacked)
    A_single_wl = H_single_wl.copy()
    A_dual_wl = H_dual_wl.copy()
    nkernels_brain = G_brain.kernel.shape[0]
    nkernels_scalp = G_scalp.kernel.shape[0]

else:
    G = None
    SB=False
    A_single_wl = Adot.copy()
    A_dual_wl = Adot_stacked.copy()
     
F_direct = None
D_direct = None
max_eig_direct = None

F_indirect = None
D_indirect = None
max_eig_indirect = None

for vv, seed_vertex in enumerate(VERTEX_LIST):
    
    print(f'\tseed vertex = {vv+1}/{len(VERTEX_LIST)}')

    for chromo in chromo_list:
                        
        print(f'\t\tchromo: {chromo}')

        all_subj_X_hrf_mag_direct = None
        all_subj_X_hrf_mag_indirect = None
        
        for subject in subject_list:
            
            print(f'\t\t\tsubject: {subject}')
            
            C_meas = C_meas_list.sel(vertex=seed_vertex, subject=subject)
            C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')

            W_direct, D_direct, F_direct, max_eig_direct = irf.calculate_W(A_dual_wl, 
                                                                        lambda_R=lambda_R, 
                                                                        alpha_meas=alpha_meas,
                                                                        alpha_spatial=alpha_spatial, 
                                                                        DIRECT=True, 
                                                                        C_meas_flag=True, 
                                                                        C_meas=C_meas, 
                                                                        D=D_direct,
                                                                        F=F_direct, 
                                                                        max_eig=max_eig_direct)
            
        
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


            # want to use a single point absorber 
            ground_truth = np.zeros( (nV_brain+nV_scalp) * 2)
        
            # generate blob of activation 
            blob_img = synthetic_hrf.build_blob_from_seed_vertex(head, vertex = seed_vertex, scale = BLOB_SIGMA)
            
            if chromo == 'HbO':
                ground_truth[:nV_brain] = blob_img
            else:
                ground_truth[nV:nV+nV_brain] = blob_img
        
            y = Adot_stacked.values @ ground_truth 
            
            b = SCALE_FACTOR / y.max()
            y = y * b 
            expected_contrast = ground_truth * b
            
            ##### DIRECT METHOD 
            X = W_direct.values @ y
            
            split = len(X)//2
        
            if sigma_brain > 0:
                X = sbf.go_from_kernel_space_to_image_space_direct(X, G)
            else:
                X = X.reshape([2, split]).T
                
            X_direct = xr.DataArray(X,
                             dims = ('vertex', 'chromo'),
                             coords = {'chromo': ['HbO', 'HbR'],
                                       'parcel': ('vertex',Adot.coords['parcel'].values),
                                       'is_brain':('vertex', Adot.coords['is_brain'].values)},
                            )

            X_noise_direct = irf.get_image_noise_posterior(A_dual_wl, 
                                                            W_direct, 
                                                            alpha_spatial=alpha_spatial, 
                                                            lambda_R=lambda_R,
                                                            DIRECT=True, 
                                                            SB=SB, 
                                                            G=G)

            ##### INDIRECT METHOD 
            y_wl0 = y[:n_chs]
            X_wl0 = W_indirect.isel(wavelength=0).values @ y_wl0
        
            y_wl1 = y[n_chs:]
            X_wl1 = W_indirect.isel(wavelength=1).values @ y_wl1
            
            if sigma_brain > 0:
                X_wl0 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl0, G)
                X_wl1 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl1, G)
                
            X_od = np.vstack([X_wl0, X_wl1]).T
            X_od = xr.DataArray(X_od, 
                             dims = ('vertex', 'wavelength'),
                             coords = {'wavelength': rec['amp'].wavelength,
                                       'parcel': ('vertex',Adot.coords['parcel'].values),
                                       'is_brain':('vertex', Adot.coords['is_brain'].values)},
                            )
        
            
            # convert to concentraiton 
            X_indirect = xr.dot(einv, X_od/units.mm, dims=["wavelength"])
                
            X_noise_indirect = irf.get_image_noise_posterior(A_single_wl, 
                                                            W_indirect, 
                                                            alpha_spatial=alpha_spatial, 
                                                            lambda_R=lambda_R,
                                                            DIRECT=False, 
                                                            SB=SB, 
                                                            G=G)
            
            if all_subj_X_hrf_mag_direct is None:
                
                all_subj_X_hrf_mag_direct = X_direct
                all_subj_X_hrf_mag_direct = all_subj_X_hrf_mag_direct.assign_coords(subj=subject)

                all_subj_X_mse_direct = X_noise_direct
                all_subj_X_mse_direct = all_subj_X_mse_direct.assign_coords(subj=subject)

                X_hrf_mag_weighted_direct = X_direct / X_noise_direct
                X_mse_inv_weighted_direct = 1 / X_noise_direct
                
                all_subj_X_hrf_mag_indirect = X_indirect
                all_subj_X_hrf_mag_indirect = all_subj_X_hrf_mag_indirect.assign_coords(subj=subject)

                all_subj_X_mse_indirect = X_noise_indirect
                all_subj_X_mse_indirect = all_subj_X_mse_indirect.assign_coords(subj=subject)

                X_hrf_mag_weighted_indirect = X_indirect / X_noise_indirect
                X_mse_inv_weighted_indirect = 1 / X_noise_indirect
                
            else:

                X_hrf_mag_tmp = X_direct.assign_coords(subj=subject)
                X_mse_tmp = X_noise_direct.assign_coords(subj=subject)

                all_subj_X_hrf_mag_direct = xr.concat([all_subj_X_hrf_mag_direct, X_hrf_mag_tmp], dim='subj')
                all_subj_X_mse_direct = xr.concat([all_subj_X_mse_direct, X_mse_tmp], dim='subj')

                X_hrf_mag_weighted_direct = X_hrf_mag_weighted_direct + X_direct / X_noise_direct
                X_mse_inv_weighted_direct = X_mse_inv_weighted_direct + 1 / X_noise_direct 
                
                X_hrf_mag_tmp = X_indirect.assign_coords(subj=subject)
                X_mse_tmp = X_noise_indirect.assign_coords(subj=subject)

                all_subj_X_hrf_mag_indirect = xr.concat([all_subj_X_hrf_mag_indirect, X_hrf_mag_tmp], dim='subj')
                all_subj_X_mse_indirect = xr.concat([all_subj_X_mse_indirect, X_mse_tmp], dim='subj')

                X_hrf_mag_weighted_indirect = X_hrf_mag_weighted_indirect + X_indirect / X_noise_indirect
                X_mse_inv_weighted_indirect = X_mse_inv_weighted_indirect + 1 / X_noise_indirect 
                
        # get the average DIRECT
        X_hrf_mag_mean_weighted_direct = X_hrf_mag_weighted_direct / X_mse_inv_weighted_direct
        X_hrf_mag_mean_direct = all_subj_X_hrf_mag_direct.mean('subj')
        X_mse_mean_within_subject_direct = 1 / X_mse_inv_weighted_direct
                            
        X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_mag_direct - X_hrf_mag_mean_weighted_direct)**2 / all_subj_X_mse_direct # X_mse_subj_tmp is weights for each sub
        X_mse_weighted_between_subjects_direct = X_mse_weighted_between_subjects_tmp.mean('subj')
        X_mse_weighted_between_subjects_direct = X_mse_weighted_between_subjects_direct * X_mse_mean_within_subject_direct # / (all_subj_X_mse**-1).mean('subj')
        X_mse_weighted_between_subjects_direct = X_mse_weighted_between_subjects_direct.pint.dequantify()
        
        # get the weighted average
        mse_btw_within_sum_subj = all_subj_X_mse_direct + X_mse_weighted_between_subjects_direct
        denom = (1/mse_btw_within_sum_subj).sum('subj')
        
        X_hrf_mag_mean_weighted_direct = (all_subj_X_hrf_mag_direct / mse_btw_within_sum_subj).sum('subj')
        X_hrf_mag_mean_weighted_direct = X_hrf_mag_mean_weighted_direct / denom
        
        mse_total = 1/denom
        mse_total = mse_total.where(np.isfinite(mse_total), np.nan)
        X_stderr_weighted_direct = np.sqrt( mse_total )

        # get the average INDIRECT
        X_hrf_mag_mean_weighted_indirect = X_hrf_mag_weighted_indirect / X_mse_inv_weighted_indirect
        X_hrf_mag_mean_indirect = all_subj_X_hrf_mag_indirect.mean('subj')
        X_mse_mean_within_subject_indirect = 1 / X_mse_inv_weighted_indirect
                            
        X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_mag_indirect - X_hrf_mag_mean_weighted_indirect)**2 / all_subj_X_mse_indirect # X_mse_subj_tmp is weights for each sub
        X_mse_weighted_between_subjects_indirect = X_mse_weighted_between_subjects_tmp.mean('subj')
        X_mse_weighted_between_subjects_indirect = X_mse_weighted_between_subjects_indirect * X_mse_mean_within_subject_indirect # / (all_subj_X_mse**-1).mean('subj')
        X_mse_weighted_between_subjects_indirect = X_mse_weighted_between_subjects_indirect.pint.dequantify()
        
        # get the weighted average
        mse_btw_within_sum_subj = all_subj_X_mse_indirect.pint.dequantify() + X_mse_weighted_between_subjects_indirect.pint.dequantify()
        denom = (1/mse_btw_within_sum_subj).sum('subj')
        
        X_hrf_mag_mean_weighted_indirect = (all_subj_X_hrf_mag_indirect / mse_btw_within_sum_subj).sum('subj')
        X_hrf_mag_mean_weighted_indirect = X_hrf_mag_mean_weighted_indirect / denom
        
        mse_total = 1/denom
        mse_total = mse_total.where(np.isfinite(mse_total), np.nan)
        X_stderr_weighted_indirect = np.sqrt( mse_total )

        # GET METRICS FOR DIRECT
        X_brain = X_hrf_mag_mean_weighted_direct[X_hrf_mag_mean_weighted_direct.is_brain.values,:]
        X_scalp = X_hrf_mag_mean_weighted_direct[~X_hrf_mag_mean_weighted_direct.is_brain.values,:]
        X_std = X_stderr_weighted_direct[:,X_stderr_weighted_direct.is_brain.values]
        
        ROI = gim.get_ROI(X_brain.sel(chromo=chromo), 0.5)
        ROI_expected = gim.get_ROI(expected_contrast, 0.5)

        P_val = X_brain.sel(chromo=chromo)[ROI].mean('vertex') # how much the chromo is reconstructed
        
        opp = [c for c in chromo_list if c != chromo][0]
        
        C_val = X_brain.sel(chromo=opp)[ROI].mean('vertex') # how much the other chromo is reconstructed
        
        P_direct.loc[chromo, seed_vertex] = P_val.values
        C_direct.loc[chromo, seed_vertex] = C_val.values
        
        if chromo == 'HbO':
            X = X_brain.sel(chromo='HbO')
            X_scalp = X_scalp.sel(chromo='HbO')
            CNR_HbO_direct.loc[seed_vertex] = X[ROI].mean() / X_std.sel(chromo='HbO')[ROI].mean()
            contrast_ratio_HbO_direct.loc[seed_vertex] = X.max() / expected_contrast.max()
            FWHM_HbO_direct.loc[seed_vertex] = gim.get_FWHM(X, head, version='weighted_mean')
            origin = head.brain.vertices[seed_vertex,:]
            localization_error_HbO_direct.loc[seed_vertex] = gim.get_localization_error(origin, X, head)
            crosstalk_brainVscalp_HbO_direct.loc[seed_vertex], _ = gim.get_crosstalk(X, X_scalp)

        # GET METRICS INDIRECT
        X_brain = X_hrf_mag_mean_weighted_indirect[:, X_hrf_mag_mean_weighted_indirect.is_brain.values]
        X_scalp = X_hrf_mag_mean_weighted_indirect[:, ~X_hrf_mag_mean_weighted_indirect.is_brain.values]
        X_std = X_stderr_weighted_indirect[:, X_stderr_weighted_indirect.is_brain.values]

        ROI = gim.get_ROI(X_brain.sel(chromo=chromo), 0.5)
        
        P_val = X_brain.sel(chromo=chromo)[ROI].mean('vertex') # how much the chromo is reconstructed
        
        opp = [c for c in chromo_list if c != chromo][0]
        
        C_val = X_brain.sel(chromo=opp)[ROI].mean('vertex') # how much the other chromo is reconstructed
        
        P_indirect.loc[chromo, seed_vertex] = P_val.values
        C_indirect.loc[chromo, seed_vertex] = C_val.values
    
        if chromo == 'HbO':
            X = X_brain.sel(chromo='HbO')
            X_scalp = X_scalp.sel(chromo='HbO')
            CNR_HbO_indirect.loc[seed_vertex] = X[ROI].mean() / X_std.sel(chromo='HbO')[ROI].mean()            
            contrast_ratio_HbO_indirect.loc[seed_vertex] = X.max() / expected_contrast.max()
            FWHM_HbO_indirect.loc[seed_vertex] = gim.get_FWHM(X, head, version='weighted_mean')
            origin = head.brain.vertices[seed_vertex,:]
            localization_error_HbO_indirect.loc[seed_vertex] = gim.get_localization_error(origin, X, head)
            crosstalk_brainVscalp_HbO_indirect.loc[seed_vertex], _ = gim.get_crosstalk(X, X_scalp)

crosstalk_HbOVHbR_direct = C_direct.sel(chromo='HbO')/P_direct.sel(chromo='HbO')
crosstalk_HbRVHbO_direct = C_direct.sel(chromo='HbR')/P_direct.sel(chromo='HbR')
crosstalk_HbOVHbR_indirect = C_indirect.sel(chromo='HbO')/P_indirect.sel(chromo='HbO')
crosstalk_HbRVHbO_indirect = C_indirect.sel(chromo='HbR')/P_indirect.sel(chromo='HbR')

RESULTS = {
           'crosstalk_HbOVHbR_direct': crosstalk_HbOVHbR_direct,
           'crosstalk_HbRVHbO_direct':  crosstalk_HbRVHbO_direct,
           'crosstalk_HbOVHbR_indirect': crosstalk_HbOVHbR_indirect,
           'crosstalk_HbRVHbO_indirect': crosstalk_HbRVHbO_indirect,
           'CNR_HbO_direct': CNR_HbO_direct,
           'CNR_HbO_indirect': CNR_HbO_indirect,
            'contrast_ratio_HbO_direct': contrast_ratio_HbO_direct,
            'contrast_ratio_HbO_indirect': contrast_ratio_HbO_indirect, 
           'FWHM_HbO_indirect': FWHM_HbO_indirect,
           'FWHM_HbO_direct': FWHM_HbO_direct,
           'localization_error_HbO_indirect':localization_error_HbO_indirect,
           'localization_error_HbO_direct':localization_error_HbO_direct,
           'crosstalk_brainVscalp_HbO_direct': crosstalk_brainVscalp_HbO_direct,
           'crosstalk_brainVscalp_HbO_indirect': crosstalk_brainVscalp_HbO_indirect,
           }
  
with open(os.path.join(SAVE_DIR, f'COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_sb-{sigma_brain}_ss-{sigma_scalp}_am-{alpha_meas}_as-{alpha_spatial}_lR-{lambda_R}_{NOISE_MODEL}_dual_wl.pkl'), 'wb') as f:
    pickle.dump(RESULTS, f)
 
print('Job Complete.')

# %%
