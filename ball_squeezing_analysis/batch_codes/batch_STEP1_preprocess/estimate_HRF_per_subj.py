
"""
estimate_HRF_per_subj.py

Preprocessing and hemodynamic response function (HRF) estimation pipeline for the
ball-squeezing fNIRS dataset as done in "Surface-Based Image Reconstruction Optimization for High-Density Functional Near Infrared Spectroscopy".
This script is the per-subject worker called by the FIG8 HRF estimation
batch submitter.

This script performs the following high-level steps for a given subject and run:

- Load raw SNIRF recordings and associated *_events.tsv stimulus files.
- Identify bad channels based on amplitude, SNR and other heuristics.
- Convert raw amplitudes to optical density (OD) and concentration.
- Optional motion correction (TDDR) and optional bandpass filtering.
- Fit a GLM to estimate the HRF (supports 'ols' and 'ar_irls' noise models).
- Save per-subject preprocessed results and HRF estimates as gzipped pickles
  under <ROOT_DIR>/derivatives/processed_data/<sub>.

Usage
-----
- Edit configuration values in the "CONFIGURE" section below (ROOT_DIR,
  NOISE_MODEL, RUN_PREPROCESS, RUN_HRF_ESTIMATION, etc.).
- Run from the command line with a single subject id (BIDS-style):

        python estimate_HRF_per_subj.py sub-618

Inputs
------
- A BIDS-like folder structure under ROOT_DIR containing subject folders with
  a `nirs` subfolder and files like
  `<sub>_task-<TASK>_run-0X_nirs.snirf` plus matching `_events.tsv` files.

  
Configurables (defaults shown)
-----------------------------
- ROOT_DIR (str): '/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids'
    - Root dataset path containing subject folders.
- RUN_PREPROCESS (bool): True
    - If True, perform preprocessing and save per-subject intermediate results.
    - Else, load previously saved preprocessed data.
- RUN_HRF_ESTIMATION (bool): True
    - If True, run the GLM-based HRF estimation step.
- SAVE_RESIDUAL (bool): False
    - If True, save GLM residuals per subject.
- NOISE_MODEL (str): 'ols'  # supported: 'ols', 'ar_irls'
    - Noise model used for GLM fitting.
    - Controls whether TDDR/bandpass (ols) or raw concentration (ar_irls) is used.
- TASK (str): 'BS'
    - Task identifier used to build file IDs.
- N_RUNS (int): 3
    - Number of runs to process per subject.

GLM configuration (constructed from globals)
------------------------------------------
- cfg_GLM (dict): keys set automatically from NOISE_MODEL and DRIFT_ORDER:
    - do_drift (bool), do_drift_legendre (bool), do_short_sep (bool),
      drift_order (int), distance_threshold (pint length), short_channel_method (str),
      noise_model (str), t_delta/t_std/t_pre/t_post (pint time values)

Dataset and pruning parameters
-----------------------------
- cfg_dataset (dict): contains 'root_dir', 'subj_ids', and 'file_ids' (built from TASK and N_RUNS).
- cfg_prune (dict): channel pruning thresholds and parameters (defaults shown in script):
    - snr_thresh: 5
    - sd_thresh: [1, 40] * mm
    - amp_thresh: [1e-3, 0.84] * V
    - perc_time_clean_thresh, sci_threshold, psp_threshold, window_length, flag_use_sci, flag_use_psp
    - channel_sel: derived from the forward-model Adot (`Adot.channel`)

- cfg_bandpass (dict): {'fmin': 0*Hz, 'fmax': 0.5*Hz}  # depends on NOISE_MODEL

- cfg_mse (dict): values used to detect/flag bad channels by MSE and amplitude thresholds.

Outputs
-------
- Per-subject gzip-compressed pickle containing a dict with keys:
  - 'hrf_per_subj' (xarray): estimated HRF per channel/time/chromophore/trial_type
  - 'hrf_mse_per_subj' (xarray): MSE of HRF estimates
  - 'bad_indices' (np.ndarray): indices of bad channels
  - saved to <ROOT_DIR>/derivatives/processed_data/<subj>/<subj>_conc_o_hrf_estimates_<NOISE_MODEL>.pkl.gz
- Optional GLM residual saved to the subject save directory when
    SAVE_RESIDUAL is True.

Dependencies
------------
Requires the project `cedalion` package and helper modules available via
the project's modules path (processing_func). Also uses xarray, numpy,
pandas and python's gzip/pickle.

Author: Laura Carlton
"""

# %% Imports
##############################################################################
import os
import gzip
import pickle
import sys
import warnings

import pandas as pd 
import numpy as np 
import xarray as xr

from cedalion import units, nirs, io
from cedalion.io.forward_model import load_Adot
import cedalion.sigproc.motion_correct as motion
from cedalion.sigproc.frequency import freq_filter

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import processing_func as pf

# Turn off all warnings
warnings.filterwarnings('ignore')

#%%
subject = str(sys.argv[1])
# subject = 'sub-618'

#%% Initial root directory and analysis parameters
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids')
RUN_PREPROCESS = True
TASK = 'BS'
RUN_HRF_ESTIMATION = True
SAVE_RESIDUAL = True
NOISE_MODEL = 'ar_irls'
N_RUNS = 3

if NOISE_MODEL == 'ols':
    DO_TDDR = True
    DO_DRIFT = True
    DO_DRIFT_LEGENDRE = False
    DRIFT_ORDER = 3
    F_MIN = 0 * units.Hz
    F_MAX = 0.5 * units.Hz
elif NOISE_MODEL == 'ar_irls':
    DO_TDDR = False
    DO_DRIFT = False
    DO_DRIFT_LEGENDRE = True
    DRIFT_ORDER = 3
    F_MAX = 0
    F_MIN = 0
else:
    print('Not a valid noise model - please select ols or ar_irls')

SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'processed_data', subject)
os.makedirs(SAVE_DIR, exist_ok=True)            

PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw', 'ICBM152')
Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))

cfg_GLM = {
    'do_drift': DO_DRIFT,
    'do_drift_legendre': DO_DRIFT_LEGENDRE,
    'do_short_sep': True,
    'drift_order' : DRIFT_ORDER,
    'distance_threshold' : 20*units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : NOISE_MODEL,
    't_delta' : 1*units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1*units.s ,  
    't_pre' : 2*units.s,
    't_post' : 10*units.s
    }

cfg_dataset = {

    'root_dir' : ROOT_DIR,
    'subj_ids' : [subject],
    'file_ids' : [f'{TASK}_run-0{i}' for i in range(1, N_RUNS+1)],
}

cfg_prune = {
    'snr_thresh' : 5, # the SNR (std/mean) of a channel. 
    'sd_thresh' : [1, 40]*units.mm, # defines the lower and upper bounds for the source-detector separation that we would like to keep
    'amp_thresh' : [1e-3, 0.84]*units.V, # define whether a channel's amplitude is within a certain range
    'perc_time_clean_thresh' : 0.6,
    'sci_threshold' : 0.6,
    'psp_threshold' : 0.1,
    'window_length' : 5 * units.s,
    'flag_use_sci' : False,
    'flag_use_psp' : False,
    'channel_sel': Adot.channel
}

cfg_bandpass = { 
    'fmin' : F_MIN,
    'fmax' : F_MAX
}

# values for manual adjustment of channel space MSE in OD
cfg_mse = {
    'mse_val_for_bad_data' : 1e1, 
    'mse_amp_thresh' : 1e-3*units.V,
    'blockaverage_val' : 0 ,
     'mse_min_thresh' : 1e-6
    }

#%% RUN PREPROCESSING

if RUN_PREPROCESS:
    print('RUNNING PREPROCESSING')
    # loop over subjects and files

    for file_idx in range(N_RUNS):
            
        filenm = f"{subject}_task-{cfg_dataset['file_ids'][file_idx]}_nirs"
        print( f"Processing  {file_idx+1} of {N_RUNS} files : {filenm}" )

        subStr = filenm.split('_')[0]
        subDir = os.path.join(cfg_dataset['root_dir'], subStr, 'nirs')
        
        file_path = os.path.join(subDir, filenm )
        records = io.read_snirf( file_path )
    
        rec = records[0]
        stim_df = pd.read_csv( file_path[:-5] + '_events.tsv', sep='\t' )
        rec.stim = stim_df        

        rec['amp'] = rec['amp'].sel(channel=cfg_prune['channel_sel'])
        rec['amp'] = rec['amp'].where( ~rec['amp'].isnull(), 1e-18 )
        rec['amp'] = rec['amp'].where( rec['amp']>0, 1e-18 )

        # if first value is 1e-18 then replace with second value
        indices = np.where(rec['amp'][:,0,0] == 1e-18)
        rec['amp'][indices[0],0,0] = rec['amp'][indices[0],0,1]
        indices = np.where(rec['amp'][:,1,0] == 1e-18)
        rec['amp'][indices[0],1,0] = rec['amp'][indices[0],1,1]
            
        rec['amp'] = rec['amp'].pint.dequantify().pint.quantify('V')
        rec, chs_pruned = pf.prune_channels(rec, cfg_prune['amp_thresh'], 
                                                cfg_prune['sd_thresh'], 
                                                cfg_prune['snr_thresh'])

        dpf = xr.DataArray(
                            [1, 1],
                            dims="wavelength",
                            coords={"wavelength": rec["amp"].wavelength},
                            )

        rec["od_o"] = nirs.int2od(rec['amp'])
        rec['od_o'].time.attrs['units'] = units.s

        if DO_TDDR:
            rec['od_o'] = motion.tddr(rec['od_o'])

        rec['od_o'] = rec['od_o'].where( ~rec['od_o'].isnull(), 1e-18 ) 

        if cfg_bandpass['fmin'] > 0 or cfg_bandpass['fmax'] > 0:
            rec["od_o"] = freq_filter(rec["od_o"], 
                                        cfg_bandpass['fmin'], 
                                        cfg_bandpass['fmax'])

        rec['conc_o'] = nirs.od2conc(rec['od_o'], rec.geo3d, dpf)

        if file_idx == 0:
            all_runs = []
            all_chs_pruned = []
            all_stims = []
        
            all_runs.append( rec )
            all_chs_pruned.append( chs_pruned )
            all_stims.append( stim_df )

        else:
            all_runs.append( rec )
            all_chs_pruned.append( chs_pruned )
            all_stims.append( stim_df )

    print("Saving data")
    geo3d = rec.geo3d
    results = {'runs': all_runs,
                'chs_pruned': all_chs_pruned,
                'stims': all_stims,
                'geo3d': geo3d
                }

    with gzip.open( os.path.join(SAVE_DIR, f'{subject}_task-{TASK}_preprocessed_results_{NOISE_MODEL}.pkl'), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL
        )
else:
    print('LOADING PREPROCESSED DATA')
    with gzip.open( os.path.join(SAVE_DIR, f'{subject}_task-{TASK}_preprocessed_results_{NOISE_MODEL}.pkl'), 'rb') as f:
        results = pickle.load(f)
    
    all_runs = results['runs']
    all_chs_pruned = results['chs_pruned']
    all_stims = results['stims']
    geo3d = results['geo3d']

#%% RUN HRF ESTIMATION

if RUN_HRF_ESTIMATION:

    wavelengths =  all_runs[0]['amp'].wavelength
    dpf = xr.DataArray(
                        [1, 1],
                        dims="wavelength",
                        coords={"wavelength": wavelengths},
                        )
                        
    REC_STR = 'conc_o'
    
    run_ts_list = [run[REC_STR] for run in all_runs]
    results, hrf_estimate, hrf_mse = pf.GLM(run_ts_list, cfg_GLM, geo3d, all_chs_pruned, all_stims)
    residual = results.sm.resid

    # reset the values for bad channels 
    amp = all_runs[0]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
    n_chs = len(amp.channel)
    idx_amp = np.where(amp < cfg_mse['mse_amp_thresh'])[0]
    idx_sat = np.where(all_chs_pruned[0] == 0.0)[0]
    bad_indices = np.unique(np.concat([idx_amp, idx_sat]))

    hrf_estimate = hrf_estimate.transpose('channel', 'time', 'chromo', 'trial_type')
    hrf_estimate = hrf_estimate - hrf_estimate.sel(time=(hrf_estimate.time < 0)).mean('time')

    hrf_mse = hrf_mse.transpose('channel', 'time', 'chromo', 'trial_type')

    hrf_per_subj = hrf_estimate.expand_dims('subj')
    hrf_per_subj = hrf_per_subj.assign_coords(subj=subject)

    hrf_mse_per_subj = hrf_mse.expand_dims('subj')
    hrf_mse_per_subj = hrf_mse_per_subj.assign_coords(subj=subject)

    print('HRF estimation complete')

    # save per subject results concentration and then image recon will take and convert to OD 
    file_path_pkl = os.path.join(SAVE_DIR, f"{subject}_task-{TASK}_{REC_STR}_hrf_estimates_{NOISE_MODEL}.pkl.gz")

    # save the individual results to a pickle file for image recon
    file = gzip.GzipFile(file_path_pkl, 'wb')

    all_results = {
                'hrf_per_subj': hrf_per_subj,  # always unweighted   - load into img recon
                'hrf_mse_per_subj': hrf_mse_per_subj, # - load into img recon
                'bad_indices': bad_indices,
            }

    file.write(pickle.dumps(all_results))
    file.close()

    if SAVE_RESIDUAL:
        file_path_pkl = os.path.join(SAVE_DIR, f"{subject}_task-{TASK}_{REC_STR}_glm_residual_{NOISE_MODEL}.pkl")

        residual = results.sm.resid
        with open(file_path_pkl, 'wb') as f:
            pickle.dump(residual, f)

    print('Saved individual HRF to ' + file_path_pkl)

print('Job Complete.')
# %%
