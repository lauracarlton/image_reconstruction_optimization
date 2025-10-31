#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For a given set of image recon parameters
- get the metrics on the subject average at each vertex
- specify the root directory, size the of the blob, scale of the HRF, list of seed vertices

Steps for user:
configure the following parameters:
    - ROOT_DIR: should point to a BIDs data folder
    - HEAD_MODEL: which atlas to use - options in cedalion are Colin27 and ICBM152
    - GLM_METHOD: which solving method was used in preprocessing of augmented data - ols or ar_irls
    - TASK: which of the tasks in the BIDS dataset was augmented 
    - BLOB_SIGMA: the standard deviation of the Gaussian blob of activation (mm)
    - SCALE_FACTOR: the amplitude of the maximum change in 850nm OD in channel space
    - WL_IDX: index of the wavelength to use for the single wavelength simulation
    - VERTEX_LIST: list of seed vertices to be used 
    - exclude_subj: any subjects IDs within the BIDs dataset to be excluded
choose the image recon parameters to test 
- select range of alpha measurement
- select range of alpha spatial 
- select range of sigma brain 
- select range of sigma scalp 

@author: lcarlton
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
from cedalion import units, nirs, xrutils, io
from cedalion.io.forward_model import load_Adot

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import image_recon_func as irf
import spatial_basis_funs as sbf
import get_image_metrics as gim  

warnings.filterwarnings('ignore')

#%% CONFIG 
ROOT_DIR = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids/"
HEAD_MODEL = 'ICBM152'
GLM_METHOD = 'ols'
TASK = 'RS'
BLOB_SIGMA = 15 * units.mm
SCALE_FACTOR = 0.02
WL_IDX = 1 # use wavelength at index 1 (850 nm according to our dataset)
VERTEX_LIST = [10089, 10453, 14673, 11323, 13685, 11702, 8337]
exclude_subj = ['sub-577']

# IMAGE RECON PARAMS TO TEST 
alpha_meas_list = [1e-4, 1, 1e4] 
alpha_spatial_list = [1e-2, 1e-3] 
sigma_brain_list = [0, 5] *units.mm 
sigma_scalp_list = [0, 10] *units.mm 

#%% SETUP DOWNSTREAM CONFIGS
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'fw', HEAD_MODEL)

dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if 'sub' in d and d not in exclude_subj]

# HEAD PARAMS
MASK_THRESHOLD = -2
position_skew = [700, 100, 0]

#%% LOAD DATA
subj_temp =  subject_list[0] + '/nirs/' + subject_list[0] + '_task-RS_run-01_nirs.snirf'
file_name = os.path.join(ROOT_DIR, subj_temp)
rec = io.read_snirf(file_name)[0]

head, parcel_dir = irf.load_head_model(with_parcels=False)
Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))

channels = Adot.channel
n_chs = len(channels)
A_fw = Adot.isel(wavelength = 1)
nV_brain = A_fw.is_brain.sum().values

#%% LOAD IN CMEAS
with open(SAVE_DIR + f"C_meas_subj_{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_{NOISE_MODEL}.pkl", 'rb') as f:
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
    
M = sbf.get_sensitivity_mask(Adot, MASK_THRESHOLD, 1)

for sigma_brain in sigma_brain_list: 
     
    if sigma_brain > 0:
        print(f'\tsigma brain = {sigma_brain.magnitude}')
        if os.path.exists(PROBE_DIR + f'/G_matrix_sigmabrain-{sigma_brain}.pkl'):
            with open(PROBE_DIR + f'/G_matrix_sigmabrain-{sigma_brain}.pkl', 'rb') as f:
                G_brain = pickle.load(f)
        else:
            brain_downsampled = sbf.downsample_mesh(head.brain.vertices, M[M.is_brain], sigma_brain*units.mm)
            G_brain = sbf.get_kernel_matrix(brain_downsampled, head.brain.vertices, sigma_brain*units.mm)
            with open(PROBE_DIR + f'/G_matrix_sigmabrain-{sigma_brain}.pkl', 'wb') as f:
                    pickle.dump(G_brain, f)

    for sigma_scalp in sigma_scalp_list:
        
        if sigma_scalp > 0 and sigma_brain > 0:
            print(f'\tsigma scalp = {sigma_scalp.magnitude}')
            if os.path.exists( PROBE_DIR + f'/G_matrix_sigmascalp-{sigma_scalp}.pkl'):
                with open(PROBE_DIR + f'/G_matrix_sigmascalp-{sigma_scalp}.pkl', 'rb') as f:
                    G_scalp = pickle.load(f)
            else:
                scalp_downsampled = sbf.downsample_mesh(head.scalp.vertices, M[~M.is_brain], sigma_scalp*units.mm)
                G_scalp = sbf.get_kernel_matrix(scalp_downsampled, head.scalp.vertices, sigma_scalp*units.mm)
                with open(PROBE_DIR + f'/G_matrix_sigmascalp-{sigma_scalp}.pkl', 'wb') as f:
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
                print(f'\t\tseed vertex = {ii+1}/{len(VERTEX_LIST)}')

                for alpha_meas in alpha_meas_list:
                    # alpha_meas = 1e5
                    print(f'\t\talpha_meas = {alpha_meas}')
                    all_subj_X_hrf_mag = None
                    
                    for subject in subject_list:
                        print(f'\t\t\tsubject: {subject}')
                        
                        #### SINGLE WAVELENGTH IMAGE
                        C_meas = C_meas_list.sel(vertex=seed_vertex, subject=subject)

                        # get the pseudoinverse
                        W_indirect, D_indirect, F_indirect = irf.calculate_W(A_single_wl, alpha_meas, alpha_spatial, DIRECT=False,
                                                     C_meas_flag=True, C_meas=C_meas, D=D_indirect, F=F_indirect)
                       
                        W_single_wl = W_indirect.sel(wavelength=850)

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
                        
                        cov_img_tmp = W_single_wl *np.sqrt(C_meas.sel(wavelength=850).values) # W is pseudo inverse  --- diagonal (faster than W C W.T)
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
with open(SAVE_DIR + f'COMPILED_METRIC_RESULTS_{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_{GLM_METHOD}_single_wl.pkl', 'wb') as f:
    pickle.dump(RESULTS, f)

# %%
