#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:48:36 2025

@author: lcarlton
"""
#%%
import os
import gzip
import pickle

import xarray as xr
from cedalion import nirs, units, io
from cedalion.io.forward_model import load_Adot

# import my own functions from a different directorys
import sys
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import image_recon_func as irf
import spatial_basis_funs as sbf

# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')

#%% set up config parameters
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids/')

dirs = os.listdir(ROOT_DIR)
excluded = ['sub-538', 'sub-549', 'sub-547'] 
subject_list = [d for d in dirs if 'sub' in d and d not in excluded]

PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw', 'ICBM152')

NOISE_MODEL = 'ar_irls'
REC_STR = 'conc_o'

C_meas_flag = True
MAG_TS_FLAG = 'mag'

T_WIN = [5, 8]
head_model = 'ICBM152'

cfg_mse = {
    'mse_val_for_bad_data' : 1e1 , 
    'mse_amp_thresh' : 1e-3*units.V,
    'blockaverage_val' : 0 ,
     'mse_min_thresh' : 1e-6
    }

#%% load head model 
head, PARCEL_DIR = irf.load_head_model('ICBM152', with_parcels=True)
Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))

#%% run image recon
"""
do the image reconstruction of each subject independently 
- this is the unweighted subject block average magnitude 
- then reconstruct their individual MSE
- then get the weighted average in image space 
- get the total standard error using between + within subject MSE 
"""

threshold = -2 # log10 absolute
wl_idx = 1

M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)

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

dpf = xr.DataArray(
                    [1, 1],
                    dims="wavelength",
                    coords={"wavelength": amp.wavelength},
                    )
E = nirs.get_extinction_coefficients('prahl', amp.wavelength)

#%%
for cfg in cfg_list:
    F = None
    D = None

    DIRECT = cfg['DIRECT']
    SB = cfg['SB']

    sigma_brain = cfg['sigma_brain']
    sigma_scalp = cfg['sigma_scalp']
    alpha_meas = cfg['alpha_meas']
    alpha_spatial = cfg['alpha_spatial']
    
    if os.path.exists( PROBE_DIR + f'G_matrix_sigmabrain-{float(sigma_brain)}.pkl') and os.path.exists( PROBE_DIR + f'G_matrix_sigmascalp-{float(sigma_scalp)}.pkl'):
        G_EXISTS = True
        with open(PROBE_DIR + f'G_matrix_sigmabrain-{float(sigma_brain)}.pkl', 'rb') as f:
            G_brain = pickle.load(f)

        with open(PROBE_DIR + f'G_matrix_sigmascalp-{float(sigma_scalp)}.pkl', 'rb') as f:
            G_scalp = pickle.load(f)

        G = {'G_brain': G_brain,
            'G_scalp': G_scalp}

    else:
        G_EXISTS = False
        G = None

    if DIRECT:
        direct_name = 'direct'
    else:
        direct_name = 'indirect'
    
    if C_meas_flag:
        Cmeas_name = 'Cmeas'
    else:
        Cmeas_name = 'noCmeas'
    
    cfg_sbf = { 'mask_threshold': -2,
                'threshold_brain': sigma_brain*units.mm,
                'threshold_scalp': sigma_scalp*units.mm,
                'sigma_brain': sigma_brain*units.mm,
                'sigma_scalp': sigma_scalp*units.mm,    
            }
    
    print(f'alpha_meas = {alpha_meas}, alpha_spatial = {alpha_spatial}, SB = {SB}, {direct_name}')

    all_trial_X_hrf = None
    all_trial_X_mse = None

    for subject in subject_list:

        SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'processed_data', 'image_space', subject)
        os.makedirs(SAVE_DIR, exist_ok=True)

        recordings = io.read_snirf(ROOT_DIR + f'{subject}/nirs/{subject}_task-BS_run-01_nirs.snirf')
        rec = recordings[0]
        geo3d = rec.geo3d
        amp = rec['amp']
        meas_list = rec._measurement_lists['amp']

        print("Loading saved data")
        with gzip.open( os.path.join(ROOT_DIR, 'derivatives', 'processed_data', f'{subject}/{subject}_{REC_STR}_hrf_estimates_{NOISE_MODEL}.pkl.gz'), 'rb') as f:
            all_results = pickle.load(f)
                
        subj_hrf = all_results['hrf_per_subj']
        subj_mse = all_results['hrf_mse_per_subj']
        bad_channels = all_results['bad_indices']

        print(f'\tCalculating subject = {subject}')

        for trial_type in subj_hrf.trial_type:
            
            print(f'\t\tGetting images for trial type = {trial_type.values}')
            
            hrf = subj_hrf.squeeze().sel(trial_type=trial_type)

            od_hrf =  nirs.conc2od(hrf, geo3d, dpf)

            mse = subj_mse.squeeze().sel(trial_type=trial_type)
            od_mse = xr.dot(E**2, mse, dim =['chromo']) * 1 * units.mm**2

            channels = od_hrf.channel
            od_mse.loc[:,channels.isel(channel=bad_channels),:] = cfg_mse['mse_val_for_bad_data']
            od_mse = xr.where(od_mse < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], od_mse)  # !!! maybe can be removed when we have the between subject mse
            od_hrf.loc[:,channels.isel(channel=bad_channels),:] = cfg_mse['blockaverage_val']

            od_mse_mag = od_mse.mean('time')

            if MAG_TS_FLAG == 'MAG':
                od_hrf = od_hrf.sel(time=slice(T_WIN[0], T_WIN[1])).mean('time')
                fname_flag = 'mag'
            else:
                fname_flag = 'ts'

            C_meas = od_mse_mag.pint.dequantify()
            C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
            
            X_hrf, W, D, F, G = irf.do_image_recon(od_hrf, head = head, Adot = Adot, C_meas_flag=C_meas_flag, C_meas = C_meas, 
                                                        wavelength = [760,850], BRAIN_ONLY = False, DIRECT=DIRECT, SB = SB, 
                                                        cfg_sbf = cfg_sbf, alpha_spatial = alpha_spatial, alpha_meas = alpha_meas,
                                                        F = F, D = D, G = G)
            
            if SB and not G_EXISTS:
                with open(PROBE_DIR + f'G_matrix_sigmabrain-{float(sigma_brain)}.pkl', 'wb') as f:
                    pickle.dump(G['G_brain'], f)

                with open(PROBE_DIR + f'G_matrix_sigmascalp-{float(sigma_scalp)}.pkl', 'wb') as f:
                    pickle.dump(G['G_scalp'], f)

                G_EXISTS = True
            
            od_mse = od_mse.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
            od_mse = od_mse.transpose('measurement', 'time')
            if MAG_TS_FLAG == 'MAG':
                template = X_hrf
            else:
                template = X_hrf.isel(time=0).squeeze()

            X_mse = irf.get_image_noise(C_meas, template, W, DIRECT=DIRECT, SB=SB, G=G)

            if all_trial_X_hrf is None:
                
                all_trial_X_hrf = X_hrf
                all_trial_X_hrf = all_trial_X_hrf.assign_coords(subject=subject)
                all_trial_X_hrf = all_trial_X_hrf.assign_coords(trial_type=trial_type)

                all_trial_X_mse = X_mse
                all_trial_X_mse = all_trial_X_mse.assign_coords(subject=subject)
                all_trial_X_mse = all_trial_X_mse.assign_coords(trial_type=trial_type)
                
            else:

                X_hrf = X_hrf.assign_coords(subject=subject)
                X_hrf = X_hrf.assign_coords(trial_type=trial_type)

                X_mse_tmp = X_mse.assign_coords(subject=subject)
                X_mse_tmp = X_mse_tmp.assign_coords(trial_type=trial_type)

                all_trial_X_hrf = xr.concat([all_trial_X_hrf, X_hrf], dim='trial_type')
                all_trial_X_mse = xr.concat([all_trial_X_mse, X_mse_tmp], dim='trial_type')

        results = {'X_hrf': all_trial_X_hrf,
                    'X_mse': all_trial_X_mse
                    }

        print(f'\t\tSaving to {SAVE_DIR}')

        if SB:
            filepath = os.path.join(SAVE_DIR, f'{subject}_image_hrf_{fname_flag}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{Cmeas_name}_{NOISE_MODEL}.pkl.gz')
        else:
            filepath = os.path.join(SAVE_DIR, f'{subject}_image_hrf_{fname_flag}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_{Cmeas_name}_{NOISE_MODEL}.pkl.gz')
        
        file = gzip.GzipFile(filepath, 'wb')
        file.write(pickle.dumps(results))
        file.close()     

    print('Job Complete')
    # %%