
# %% Imports
##############################################################################
#%matplotlib widget

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
import image_recon_func as irf
import processing_func as pf

# Turn off all warnings
warnings.filterwarnings('ignore')

#%%
subject = str(sys.argv[1])
# subject = 'sub-618'

#%% Initial root directory and analysis parameters
ROOT_DIR = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids/"
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'processed_data', subject)
os.makedirs(SAVE_DIR, exist_ok=True)            

PROBE_DIR = ROOT_DIR + 'derivatives/cedalion/fw/ICBM152/'
Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))

RUN_PREPROCESS = True
RUN_HRF_ESTIMATION = True
SAVE_RESIDUAL = True
NOISE_MODEL = 'ar_irls'

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
    'file_ids' : ['BS_run-01', 'BS_run-02', 'BS_run-03'],
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

# if block averaging on OD:
cfg_mse = {
    'mse_val_for_bad_data' : 1e1, 
    'mse_amp_thresh' : 1e-3*units.V,
    'blockaverage_val' : 0 ,
     'mse_min_thresh' : 1e-6
    }

n_files_per_subject = len(cfg_dataset['file_ids'])

#%% RUN PREPROCESSING
# make sure derivatives folders exist
der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives')
if not os.path.exists(der_dir):
    os.makedirs(der_dir)
der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots')
if not os.path.exists(der_dir):
    os.makedirs(der_dir)
der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data')
if not os.path.exists(der_dir):
    os.makedirs(der_dir)

if RUN_PREPROCESS:
    print('RUNNING PREPROCESSING')
    # loop over subjects and files

    for file_idx in range(n_files_per_subject):
            
        filenm = f"{subject}_task-{cfg_dataset['file_ids'][file_idx]}_nirs"
        print( f"Processing  {file_idx+1} of {n_files_per_subject} files : {filenm}" )
        
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

    with gzip.open( os.path.join(SAVE_DIR, f'{subject}_preprocessed_results_{NOISE_MODEL}.pkl'), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL
        )
else:
    print('LOADING PREPROCESSED DATA')
    with gzip.open( os.path.join(SAVE_DIR, f'{subject}_preprocessed_results_{NOISE_MODEL}.pkl'), 'rb') as f:
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
    file_path_pkl = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', subject,
                                    f"{subject}_{REC_STR}_hrf_estimates_{NOISE_MODEL}.pkl.gz")

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
        file_path_pkl = os.path.join(SAVE_DIR, f"{subject}_{REC_STR}_glm_residual_{NOISE_MODEL}.pkl")

        residual = results.sm.resid
        with open(file_path_pkl, 'wb') as f:
            pickle.dump(residual, f)

    print('Saved individual HRF to ' + file_path_pkl)

print('Job Complete.')
# %%
