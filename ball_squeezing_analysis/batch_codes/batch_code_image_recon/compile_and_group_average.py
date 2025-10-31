
#%%
import os 
import pickle
import gzip
import sys
import xarray as xr
import numpy as np 
from cedalion import units
from cedalion.io.forward_model import load_Adot

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import image_recon_func as irf
import processing_func as pf
import spatial_basis_funs as sbf

FLAG_DO_SPATIAL_SMOOTHING = True
FLAG_USE_ONLY_SENSITIVE = True

ROOT_DIR = os.path.join('/', 'projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids')
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'processed_data', 'image_space')
PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw', 'ICBM152')

NOISE_MODEL = 'ar_irls'
fname_flag = 'ts'
sigma_smoothing = 30 * units.mm

if FLAG_DO_SPATIAL_SMOOTHING: 
    head, PARCEL_DIR = irf.load_head_model('ICBM152', with_parcels=True)
    Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))
    sensitivity_mask = sbf.get_sensitivity_mask(Adot.sel(vertex=Adot.is_brain.values))

    # get MNI coordinates
    V_ijk = head.brain.mesh.vertices  # shape (N,3)
    M = head.t_ijk2ras.values  # shape (4,4)
    V_h = np.c_[V_ijk, np.ones((V_ijk.shape[0], 1))]  # (N,4)
    V_ras_brain = (V_h @ M.T)[:, :3]

    if FLAG_USE_ONLY_SENSITIVE:
        V_ras = V_ras_brain[sensitivity_mask, :]
    else:
        V_ras = V_ras_brain
    
    W = pf.get_spatial_smoothing_kernel(V_ras, sigma_smoothing.magnitude)
    smoothing_name = f'_smoothing-{sigma_smoothing.magnitude}'
else:
    smoothing_name = ''

dirs = os.listdir(ROOT_DIR)

excluded = ['sub-538', 'sub-549', 'sub-547'] 
subject_list = [d for d in dirs if 'sub' in d and d not in excluded]

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

trial_types = ['right', 'left']

#%%
for cfg in cfg_list[:1]:

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

    all_trial_all_subj_X_hrf_ts = None
    print(f'alpha_meas = {alpha_meas}, alpha_spatial = {alpha_spatial}, SB = {SB}, {direct_name}')

    for trial_type in trial_types:          
        all_subj_X_hrf_ts = []
        all_subj_X_mse = []
        print(f'\ttrial_type - {trial_type}')

        for subj in subject_list:
            # print(f'\t\t{subj}')
            folderpath = os.path.join(SAVE_DIR, subj)
            if SB:
                filepath = os.path.join(folderpath, f'{subj}_image_hrf_{fname_flag}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_Cmeas_{NOISE_MODEL}.pkl.gz')
            else:
                filepath = os.path.join(folderpath, f'{subj}_image_hrf_{fname_flag}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_Cmeas_{NOISE_MODEL}.pkl.gz')
            
            with gzip.open(filepath, 'rb') as f:
                results = pickle.load(f)
            
            X_hrf_ts = results['X_hrf'].sel(trial_type=trial_type)
            X_mse = results['X_mse'].sel(trial_type=trial_type)
            all_subj_X_hrf_ts.append(X_hrf_ts)
            all_subj_X_mse.append(X_mse)
        
        all_subj_X_hrf_ts = xr.concat(all_subj_X_hrf_ts, dim='subj')
        all_subj_X_mse = xr.concat(all_subj_X_mse, dim='subj')

        if FLAG_DO_SPATIAL_SMOOTHING:
            all_subj_X_hrf_ts = all_subj_X_hrf_ts.transpose('subj', 'chromo', 'vertex', 'time')
            all_subj_X_mse = all_subj_X_mse.transpose('subj', 'chromo', 'vertex')

            all_subj_X_hrf_ts_orig = all_subj_X_hrf_ts.copy()
            all_subj_X_mse_orig = all_subj_X_mse.copy()

            all_subj_X_hrf_ts = all_subj_X_hrf_ts_orig.sel(vertex = Adot.is_brain.values)
            all_subj_X_mse = all_subj_X_mse_orig.sel(vertex = Adot.is_brain.values)

            if FLAG_USE_ONLY_SENSITIVE:
                all_subj_X_hrf_ts = all_subj_X_hrf_ts.sel(vertex=sensitivity_mask.values)
                all_subj_X_mse = all_subj_X_mse.sel(vertex=sensitivity_mask.values)
            
            H_global = pf.compute_Hglobal_from_PCA(all_subj_X_hrf_ts, all_subj_X_mse, W)
            all_subj_X_hrf_ts_new = all_subj_X_hrf_ts - H_global
            all_subj_X_hrf_ts = all_subj_X_hrf_ts_new.copy()
        
        X_hrf_ts_mean = all_subj_X_hrf_ts.mean('subj', skipna=True)

        all_subj_X_hrf_ts_tmp = all_subj_X_hrf_ts.where(~np.isnan(all_subj_X_hrf_ts), drop=True)
        all_subj_X_mse_tmp = all_subj_X_mse.where(~np.isnan(all_subj_X_mse), drop=True)

        X_hrf_ts_mean_weighted = (all_subj_X_hrf_ts_tmp/all_subj_X_mse_tmp).sum('subj') / (1/all_subj_X_mse).sum('subj')
        
        X_mse_mean_within_subject = 1 / (1/all_subj_X_mse_tmp).sum('subj')
        X_mse_mean_within_subject = X_mse_mean_within_subject.assign_coords({'trial_type': trial_type})
            
        X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_ts_tmp - X_hrf_ts_mean_weighted)**2  
        X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp / all_subj_X_mse_tmp

        X_mse_weighted_between_subjects = X_mse_weighted_between_subjects.mean('subj') * X_mse_mean_within_subject # normalized by the within subject variances as weights
    
        X_mse_weighted_between_subjects = X_mse_weighted_between_subjects.pint.dequantify()
    
        X_mse_btw_within_sum_subj = all_subj_X_mse_tmp + X_mse_weighted_between_subjects
        denom = (1/X_mse_btw_within_sum_subj).sum('subj')
        
        X_hrf_ts_mean_weighted = (all_subj_X_hrf_ts_tmp / X_mse_btw_within_sum_subj).sum('subj')
        X_hrf_ts_mean_weighted = X_hrf_ts_mean_weighted / denom
        
        mse_total = 1/denom
    
        X_stderr_weighted = np.sqrt( mse_total )
        X_tstat = X_hrf_ts_mean_weighted / X_stderr_weighted
    
        if FLAG_DO_SPATIAL_SMOOTHING:

            if FLAG_USE_ONLY_SENSITIVE:
                template = xr.full_like(all_subj_X_hrf_ts_orig, np.nan)
                full_mask = xr.zeros_like(Adot.is_brain, dtype=bool)
                full_mask.loc[dict(vertex=Adot.is_brain.vertex[Adot.is_brain])] = sensitivity_mask

                all_subj_X_hrf_ts_tmp = template.copy()
                all_subj_X_hrf_ts_tmp.loc[dict(vertex=full_mask)] = all_subj_X_hrf_ts
                all_subj_X_hrf_ts = all_subj_X_hrf_ts_tmp.copy()

                template = xr.full_like(all_subj_X_hrf_ts_orig.isel(subj=0).squeeze(), np.nan)

                X_hrf_ts_mean_weighted_tmp = template.copy()
                X_hrf_ts_mean_weighted_tmp.loc[dict(vertex=full_mask)] = X_hrf_ts_mean_weighted
                X_hrf_ts_mean_weighted = X_hrf_ts_mean_weighted_tmp.copy()

                X_tstat_tmp = template.copy()
                X_tstat_tmp.loc[dict(vertex=full_mask)] = X_tstat
                X_tstat = X_tstat_tmp.copy()

                X_stderr_weighted_tmp = template.copy()
                X_stderr_weighted_tmp.loc[dict(vertex=full_mask)] = X_stderr_weighted
                X_stderr_weighted = X_stderr_weighted_tmp.copy()

                X_mse_weighted_between_subjects_tmp = template.copy()
                X_mse_weighted_between_subjects_tmp.loc[dict(vertex=full_mask)] = X_mse_weighted_between_subjects
                X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp.copy()

                X_mse_mean_within_subject_tmp = template.copy()
                X_mse_mean_within_subject_tmp.loc[dict(vertex=full_mask)] = X_mse_mean_within_subject
                X_mse_mean_within_subject = X_mse_mean_within_subject_tmp.copy()

            else:
                all_subj_X_hrf_ts = all_subj_X_hrf_ts_orig.copy()
                all_subj_X_hrf_ts.loc[dict(vertex=Adot.is_brain.values)] = all_subj_X_hrf_ts_new.copy()

        
        if all_trial_all_subj_X_hrf_ts is None:

            all_trial_all_subj_X_hrf_ts = all_subj_X_hrf_ts
            all_trial_all_subj_X_mse = all_subj_X_mse_orig
            
            all_trial_X_hrf_ts = X_hrf_ts_mean
            all_trial_X_hrf_ts_weighted = X_hrf_ts_mean_weighted
            all_trial_X_stderr = X_stderr_weighted
            all_trial_X_tstat = X_tstat
            all_trial_X_mse_between = X_mse_weighted_between_subjects
            all_trial_X_mse_within = X_mse_mean_within_subject
        else:
    
            all_trial_all_subj_X_hrf_ts = xr.concat([all_trial_all_subj_X_hrf_ts, all_subj_X_hrf_ts], dim='trial_type')
            all_trial_all_subj_X_mse = xr.concat([all_trial_all_subj_X_mse, all_subj_X_mse_orig], dim='trial_type')
            
            all_trial_X_hrf_ts = xr.concat([all_trial_X_hrf_ts, X_hrf_ts_mean], dim='trial_type')
            all_trial_X_hrf_ts_weighted = xr.concat([all_trial_X_hrf_ts_weighted, X_hrf_ts_mean_weighted], dim='trial_type')
            all_trial_X_stderr = xr.concat([all_trial_X_stderr, X_stderr_weighted], dim='trial_type')
            all_trial_X_tstat = xr.concat([all_trial_X_tstat, X_tstat], dim='trial_type')
            all_trial_X_mse_between = xr.concat([all_trial_X_mse_between, X_mse_weighted_between_subjects], dim='trial_type')
            all_trial_X_mse_within = xr.concat([all_trial_X_mse_within, X_mse_mean_within_subject], dim='trial_type')
    
    results = {'X_hrf_ts': all_trial_all_subj_X_hrf_ts,
               'X_mse': all_trial_all_subj_X_mse,
                'X_std_err': all_trial_X_stderr,
                'X_tstat': all_trial_X_tstat,
                'X_mse_between': all_trial_X_mse_between,
                'X_hrf_ts_mean': all_trial_X_hrf_ts,
                'X_hrf_ts_weighted': all_trial_X_hrf_ts_weighted,
                'X_mse_within': all_trial_X_mse_within
               }

    if SB:
        filepath = os.path.join(SAVE_DIR, f'image_hrf_{fname_flag}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_Cmeas_{NOISE_MODEL}{smoothing_name}.pkl.gz')
    else:
        filepath = os.path.join(SAVE_DIR, f'image_hrf_{fname_flag}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_Cmeas_{NOISE_MODEL}{smoothing_name}.pkl.gz')
   
    print(f'Saving to {filepath}')
    file = gzip.GzipFile(filepath, 'wb')
    file.write(pickle.dumps(results))
    file.close()     

# %%
