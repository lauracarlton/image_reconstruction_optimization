#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
be explicit about assumptions
- include the script that was used for vertex selection
- be clear what the HRF parameters are and the default values

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

#%% SETUP CONFIGS
ROOT_DIR = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids/"
HEAD_MODEL = 'ICBM152'
GLM_METHOD = 'ols'
TASK = 'RS'
BLOB_SIGMA = 15 * units.mm
SCALE_FACTOR = 0.02
VERTEX_LIST = [10089, 10453, 14673, 11323, 13685, 11702, 8337]
exclude_subj = ['sub-577']

# IMAGE RECON PARAMS TO TEST 
alpha_meas_list = [1e-4, 1, 1e4] #[10 ** i for i in range(-6, 6)]
alpha_spatial_list = [1e-2, 1e-3] 
sigma_brain_list = [0, 1]*units.mm 
sigma_scalp_list = [0, 5]*units.mm

#%% SETUP DOWNSTREAM CONFIGS
CMEAS_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
SAVE_DIR = os.path.join(CMEAS_DIR, 'batch_results')
PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw', HEAD_MODEL)

os.makedirs(SAVE_DIR, exist_ok=True)

wl_idx = 1
mask_threshold = -2
chromo_list = ['HbO', 'HbR']

dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if 'sub' in d and d not in exclude_subj]

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

ec = nirs.get_extinction_coefficients("prahl", amp.wavelength)
einv = xrutils.pinv(ec)

#%% GET THE IMAGE METRICS 
with open(SAVE_DIR + f"C_meas_subj_{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_{GLM_METHOD}.pkl", 'rb') as f:
    C_meas_list = pickle.load(f)
    
#%% GET THE METRICS FOR ALL VERTICES

FWHM_HbO_direct = xr.DataArray(np.zeros([len(alpha_meas_list), len(alpha_spatial_list), len(sigma_brain_list), len(sigma_scalp_list), len(VERTEX_LIST)]),
                      dims = ['alpha_meas', 'alpha_spatial', 'sigma_brain', 'sigma_scalp', 'vertex'],
                      coords = {'alpha_meas': alpha_meas_list,
                                'alpha_spatial': alpha_spatial_list,
                                'sigma_brain': sigma_brain_list,
                                'sigma_scalp': sigma_scalp_list,
                                'vertex': VERTEX_LIST,
                                 })
FWHM_HbO_indirect = FWHM_HbO_direct.copy()

CNR_HbO_direct = FWHM_HbO_direct.copy()
CNR_HbO_indirect = FWHM_HbO_direct.copy()

crosstalk_brainVscalp_HbO_direct = FWHM_HbO_direct.copy()
crosstalk_brainVscalp_HbO_indirect = FWHM_HbO_direct.copy()

localization_error_HbO_direct = FWHM_HbO_direct.copy()
localization_error_HbO_indirect = FWHM_HbO_direct.copy()

contrast_ratio_HbO_direct = FWHM_HbO_direct.copy()
contrast_ratio_HbO_indirect = FWHM_HbO_direct.copy()

P_direct = xr.DataArray(np.zeros([len(alpha_meas_list), len(alpha_spatial_list), len(chromo_list), len(sigma_brain_list), len(sigma_scalp_list), len(VERTEX_LIST)]),
                      dims = ['alpha_meas', 'alpha_spatial', 'chromo', 'sigma_brain', 'sigma_scalp', 'vertex'],
                      coords = {'alpha_meas': alpha_meas_list,
                                'alpha_spatial': alpha_spatial_list,
                                'chromo': chromo_list,
                                'sigma_brain': sigma_brain_list,
                                'sigma_scalp': sigma_scalp_list,
                                'vertex': VERTEX_LIST,
                                 })
C_direct = P_direct.copy()
C_indirect = P_direct.copy()
P_indirect = P_direct.copy()

M = sbf.get_sensitivity_mask(Adot, mask_threshold, wl_idx)

for sigma_brain in sigma_brain_list: 
     
    if sigma_brain > 0:
         print(f'sigma brain = {sigma_brain.magnitude}')
         brain_downsampled = sbf.downsample_mesh(head.brain.vertices, M[M.is_brain], sigma_brain)
         G_brain = sbf.get_kernel_matrix(brain_downsampled, head.brain.vertices, sigma_brain)

    for sigma_scalp in sigma_scalp_list:
        
        if sigma_scalp > 0 and sigma_brain > 0:
            print(f'sigma scalp = {sigma_scalp.magnitude}')
            scalp_downsampled = sbf.downsample_mesh(head.scalp.vertices, M[~M.is_brain], sigma_scalp)
            G_scalp = sbf.get_kernel_matrix(scalp_downsampled, head.scalp.vertices, sigma_scalp)

            G = {'G_brain': G_brain,
                 'G_scalp': G_scalp}
    
            H_single_wl = sbf.get_H(G, Adot)
            H_dual_wl = sbf.get_H_stacked(G, Adot_stacked)
            A_single_wl = H_single_wl.copy()
            A_dual_wl = H_dual_wl.copy()
            nkernels_brain = G_brain.kernel.shape[0]
            nkernels_scalp = G_scalp.kernel.shape[0]
            
        elif sigma_scalp == 0 and sigma_brain ==0:
            G = None
            A_single_wl = Adot.copy()
            A_dual_wl = Adot_stacked.copy()
            
        else:
            continue

        for alpha_spatial in alpha_spatial_list:
           
            print(f'\talpha_spatial = {alpha_spatial}')
            
            F_direct = None
            D_direct = None
            F_indirect = None
            D_indirect = None

            for ii, seed_vertex in enumerate(VERTEX_LIST):
                print(f'\t\tseed vertex = {ii+1}/{len(VERTEX_LIST)}')

                for alpha_meas in alpha_meas_list:
                
                    print(f'\t\t\t\talpha_meas = {alpha_meas}')
                    
                    for chromo in chromo_list:
                        
                        print(f'\t\t\t\t\tchromo: {chromo}')
                        all_subj_X_hrf_mag_direct = None
                        all_subj_X_hrf_mag_indirect = None
                        
                        for subject in subject_list:
                            
                            print(f'\t\t\t\t\t\tsubject: {subject}')
                            
                            C_meas = C_meas_list.sel(vertex=seed_vertex, subject=subject)
                            C_meas_dir = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
    
                            W_direct, D_direct, F_direct = irf.calculate_W(A_dual_wl, alpha_meas, alpha_spatial, DIRECT=True,
                                                       C_meas_flag=True, C_meas=C_meas_dir, D=D_direct, F=F_direct)
                            
                            W_indirect, D_indirect, F_indirect = irf.calculate_W(A_single_wl, alpha_meas, alpha_spatial, DIRECT=False,
                                                         C_meas_flag=True, C_meas=C_meas, D=D_indirect, F=F_indirect)
                           
                            # want to use a single point absorber 
                            ground_truth = np.zeros( (nV_brain+nV_scalp) * 2)
                        
                            # generate blob of activation 
                            blob_img = synthetic_hrf.build_blob_from_seed_vertex(head, vertex = seed_vertex, scale = BLOB_SIGMA * units.mm)
                            
                            if chromo == 'HbO':
                                ground_truth[:nV_brain] = blob_img
                            else:
                                ground_truth[nvertices:nvertices+nV_brain] = blob_img
                        
                            ##### DIRECT METHOD 
                            y = Adot_stacked.values @ ground_truth 

                            b = SCALE_FACTOR / y.max()
                            y = y * b 
                            expected_contrast = ground_truth * b    

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
                        
                            if sigma_brain > 0:
                                X_noise_direct = irf.get_image_noise(C_meas_dir, X_direct, W_direct, SB=True, DIRECT=True, G=G)
                            else:                                    
                                X_noise_direct = irf.get_image_noise(C_meas_dir, X_direct, W_direct, SB=False, DIRECT=True, G=None)
                

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
                                
                            if sigma_brain > 0:
                                X_noise_indirect = irf.get_image_noise(C_meas_dir, X_indirect, W_indirect, SB=True, DIRECT=False, G=G)
                            else:
                                X_noise_indirect = irf.get_image_noise(C_meas_dir, X_indirect, W_indirect, SB=False, DIRECT=False, G=None)
                                              
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
                        mse_btw_within_sum_subj = all_subj_X_mse_indirect + X_mse_weighted_between_subjects_indirect
                        denom = (1/mse_btw_within_sum_subj).sum('subj')
                        
                        X_hrf_mag_mean_weighted_indirect = (all_subj_X_hrf_mag_indirect / mse_btw_within_sum_subj).sum('subj')
                        X_hrf_mag_mean_weighted_indirect = X_hrf_mag_mean_weighted_indirect / denom
                        
                        mse_total = 1/denom
                        mse_total = mse_total.where(np.isfinite(mse_total), np.nan)
                        X_stderr_weighted_indirect = np.sqrt( mse_total )

                        # GET METRICS FOR DIRECT
                        X_brain = X_hrf_mag_mean_weighted_direct[X_hrf_mag_mean_weighted_direct.is_brain.values,:]
                        X_scalp = X_hrf_mag_mean_weighted_direct[~X_hrf_mag_mean_weighted_direct.is_brain.values,:]
                        ROI = gim.get_ROI(X_brain.sel(chromo=chromo), 0.5)
                        
                        P_val = X_brain.sel(chromo=chromo)[ROI].mean('vertex') # how much the chromo is reconstructed
                        
                        opp = [c for c in chromo_list if c != chromo][0]
                        
                        C_val = X_brain.sel(chromo=opp)[ROI].mean('vertex') # how much the other chromo is reconstructed
                        
                        P_direct.loc[alpha_meas, alpha_spatial, chromo, sigma_brain, sigma_scalp, seed_vertex] = P_val.values
                        C_direct.loc[alpha_meas, alpha_spatial, chromo, sigma_brain, sigma_scalp, seed_vertex] = C_val.values
                        
                        if chromo == 'HbO':
                            X = X_brain.sel(chromo='HbO')
                            X_scalp = X_scalp.sel(chromo='HbO')
                            CNR_HbO_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = X[ROI].mean() / X_stderr_weighted_direct.sel(chromo='HbO')[ROI].mean()
                            FWHM_HbO_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = gim.get_FWHM(X, head, version='weighted_mean')
                            origin = head.brain.vertices[seed_vertex,:]
                            localization_error_HbO_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = gim.get_localization_error(origin, X, head)
                            crosstalk_brainVscalp_HbO_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex], _ = gim.get_crosstalk(X, X_scalp)
                            contrast_ratio_HbO_direct.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = X.max() / expected_contrast.max()
                             
                        # GET METRICS INDIRECT
                        X_brain = X_hrf_mag_mean_weighted_indirect[:, X_hrf_mag_mean_weighted_indirect.is_brain.values]
                        X_scalp = X_hrf_mag_mean_weighted_indirect[:, ~X_hrf_mag_mean_weighted_indirect.is_brain.values]
                        ROI = gim.get_ROI(X_brain.sel(chromo=chromo), 0.5)
                        
                        P_val = X_brain.sel(chromo=chromo)[ROI].mean('vertex') # how much the chromo is reconstructed
                        
                        opp = [c for c in chromo_list if c != chromo][0]
                        
                        C_val = X_brain.sel(chromo=opp)[ROI].mean('vertex') # how much the other chromo is reconstructed
                        
                        P_indirect.loc[alpha_meas, alpha_spatial, chromo, sigma_brain, sigma_scalp, seed_vertex] = P_val.values
                        C_indirect.loc[alpha_meas, alpha_spatial, chromo, sigma_brain, sigma_scalp, seed_vertex] = C_val.values
                    
                        if chromo == 'HbO':
                            X = X_brain.sel(chromo='HbO')
                            X_scalp = X_scalp.sel(chromo='HbO')
                            CNR_HbO_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = X[ROI].mean() / X_stderr_weighted_indirect.sel(chromo='HbO')[ROI].mean()
                            FWHM_HbO_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = gim.get_FWHM(X, head, version='weighted_mean')
                            localization_error_HbO_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = gim.get_localization_error(origin, X, head)
                            crosstalk_brainVscalp_HbO_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex], _ = gim.get_crosstalk(X, X_scalp)
                            contrast_ratio_HbO_indirect.loc[alpha_meas, alpha_spatial, sigma_brain, sigma_scalp, seed_vertex] = X.max() / expected_contrast.max()

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
           'FWHM_HbO_indirect': FWHM_HbO_indirect,
           'FWHM_HbO_direct': FWHM_HbO_direct,
           'localization_error_HbO_indirect':localization_error_HbO_indirect,
           'localization_error_HbO_direct':localization_error_HbO_direct,
           'crosstalk_brainVscalp_HbO_direct': crosstalk_brainVscalp_HbO_direct,
           'crosstalk_brainVscalp_HbO_indirect': crosstalk_brainVscalp_HbO_indirect,
           'contrast_ratio_HbO_direct': contrast_ratio_HbO_direct,
           'contrast_ratio_HbO_indirect': contrast_ratio_HbO_indirect
    }

with open(SAVE_DIR + f'COMPILED_METRIC_RESULTS_{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_{GLM_METHOD}_dual_wl.pkl', 'wb') as f:
    pickle.dump(RESULTS, f)
