#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG5&6_STEP1_get_measurement_variance.py

Measurement variance estimation for augmented simulations. This script processes a BIDS formatted fNIRS dataset
and by augmenting data with a synthetic HRF and estimating the measurement variance using a GLM approach.

Usage
-----
Edit the CONFIG section (ROOT_DIR, NOISE_MODEL, etc.) then run::

    python FIG5&6_STEP1_get_measurement_variance.py

Inputs
------
- A BIDS-like folder structure under ROOT_DIR containing subject folders with
  a `nirs` subfolder and SNIRF files.
- Forward model file (Adot.nc) for channel selection and sensitivity matrix.

Configurables (defaults shown)
-----------------------------
Data Storage Parameters:
- ROOT_DIR (str): 
    - Root dataset path pointing to BIDS directory containing subject folders.
- EXCLUDED (list[str]): ['sub-577']
    - Subject IDs to skip during processing.
- TASK (str): 'RS'
    - Task identifier used to build file IDs.

Head Model Parameters:
- HEAD_MODEL (str): 'ICBM152'
    - Head model used.
- VERTEX_LIST (list[int]): [10089, 10453, 14673, 11323, 13685, 11702, 8337]
    - List of seed vertex indices for synthetic activation generation.

HRF Parameters:
- BLOB_SIGMA (pint Quantity): 15 * units.mm
    - Standard deviation of the Gaussian used for spatial activation blob.
- TRANGE_HRF (list[float]): [0, 15]
    - Temporal window in seconds used to define the HRF.
- STIM_DUR (float): 5
    - Length in seconds of the boxcar function to convolve with canonical HRF.
- SCALE_FACTOR (float): 0.02
    - Scale for the HRF in OD - becomes the maximum OD in the larger wavelength index.
- T_WIN (list[float]): [4, 7]
    - Window in seconds over which to average the peak of the HRF.
- PARAMS_BASIS (list[float]): [0.1000, 3.0000, 1.8000, 3.0000]
    - Parameters for tau and sigma for the modified gamma function for each chromophore.

Preprocessing Parameters:
- NOISE_MODEL (str): 'ols'  # supported: 'ols', 'ar_irls'
    - Noise model type for GLM fitting. Controls preprocessing pipeline.
- D_RANGE (list[float]): [1e-3, 0.84]
    - Mean signal amplitude minimum and maximum for channel flagging.
- SNR_THRESH (float): 5
    - Signal-to-noise ratio threshold for channel quality control.

GLM Configuration (constructed from NOISE_MODEL):
- cfg_GLM (dict): keys set automatically from NOISE_MODEL:
    - do_drift (bool), do_drift_legendre (bool), do_short_sep (bool),
      drift_order (int), distance_threshold (pint length), short_channel_method (str),
      noise_model (str), t_delta/t_std/t_pre/t_post (pint time values)

MSE Configuration:
- cfg_mse (dict): values used to detect/flag bad channels by MSE and amplitude:
    - mse_val_for_bad_data: 1e1
    - mse_amp_thresh: 1e-3 * units.V
    - blockaverage_val: 0
    - mse_min_thresh: 1e-6

Outputs
-------
- Variance estimates (C_meas) saved as gzipped pickle file under
  <ROOT_DIR>/derivatives/cedalion/augmented_data/ with filename:
  C_meas_subj_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_{GLM_METHOD}.pkl
  Contains measurement variance estimates for use in generating synthetic 
  measurements with realistic noise profiles.

Dependencies
------------
- cedalion, xarray, numpy, pandas, matplotlib, gzip, pickle, modules/image_recon_func,
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

import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt

from cedalion import io, nirs, units
import cedalion.sim.synthetic_hrf as synthetic_hrf
import cedalion.sigproc.motion_correct as motion 
from cedalion.sigproc.frequency import freq_filter
from cedalion.io.forward_model import load_Adot

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import image_recon_func as irf
import processing_func as pf

warnings.filterwarnings('ignore')

#%%
subject = sys.argv[1]
# subject = 'sub-618'

#%% CONFIG 
# DATA STORAGE PARAMS
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids_v2')
TASK = "RS"

# HEAD PARAMS
HEAD_MODEL = 'ICBM152'
VERTEX_LIST = [10089, 10453, 14673, 11323, 13685, 11702, 8337]

# HRF PARAMS
BLOB_SIGMA = 15*units.mm # standard deviation of the Gaussian used for the spatial activation
TRANGE_HRF = [0, 15] # temporal window used to define the HRF
STIM_DUR = 5 # length in seconds of the boxcar function to convolve with the canonical HRF
SCALE_FACTOR = 0.02 # scale for the HRF in OD - becomes the maximum OD in the larger wavelength index
T_WIN = [4, 7] # window in seconds over which to average the peak of the HRF
PARAMS_BASIS  = [0.1000, 3.0000, 1.8000, 3.0000] # Parameters for tau and sigma for the modified
                                                # gamma function for each chromophore.


# PREPROCESSING PARAMS
NOISE_MODEL = 'ar_irls' # can select ols for ordinary least squares or ar_irls for autoregressive model
# channel flagging params
D_RANGE = [1e-3, 0.84] # mean signal amplitude min and max
SNR_THRESH = 5 # signal to noise ratio threshold

#%% SETUP DOWNSTREAM CONFIGURABLES
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
os.makedirs(SAVE_DIR, exist_ok=True)

PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'fw', 'probe')

cmap = plt.cm.get_cmap("jet", 256)

if NOISE_MODEL == 'ols':
    DO_TDDR = True
    DO_DRIFT = True
    DO_DRIFT_LEGENDRE = False
    DRIFT_ORDER = 3
elif NOISE_MODEL == 'ar_irls':
    DO_TDDR = False
    DO_DRIFT = False
    DO_DRIFT_LEGENDRE = True
    DRIFT_ORDER = 3
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
    't_post' : 15*units.s
    }

cfg_mse = { # values used to re-scale the MSE values for noisy channels in OD
    'mse_val_for_bad_data' : 1e1, 
    'mse_amp_thresh' : 1e-3*units.V,
    'blockaverage_val' : 0 ,
    'mse_min_thresh': 1e-6
    }

#%% DEFINE THE SYNTHETIC HRF
# load in a reference snirf file
subj_temp =  f'{subject}/nirs/{subject}_task-{TASK}_run-01_nirs.snirf'
file_name = os.path.join(ROOT_DIR, subj_temp)
rec = io.read_snirf(file_name)[0]

# load in the head model
head, parcel_dir = irf.load_head_model(with_parcels=False)
Adot = load_Adot(os.path.join(PROBE_DIR, 'Adot.nc'))
channel = Adot.channel

E = nirs.get_extinction_coefficients("prahl", Adot.wavelength)
dpf = xr.DataArray(
                    [1, 1],
                    dims="wavelength",
                    coords={"wavelength": Adot.wavelength},
                    )

# define the HRF
dt = 1/rec['amp'].cd.sampling_rate
time_hrf = np.arange(TRANGE_HRF[0], TRANGE_HRF[1], dt)
time_xr = xr.DataArray(time_hrf,
                      dims=['time'],
                      coords = {'time':time_hrf},
                      attrs={'units':'seconds'})

tbasis = synthetic_hrf.generate_hrf(time_xr, STIM_DUR,
                                    params_basis = PARAMS_BASIS) 

#%% AUGMENT DATA AND GET THE VARIANCE 
# initialize the array 
meas = rec['amp'].stack(measurement = ['channel', 'wavelength']).sortby('wavelength').measurement
C_meas_list = xr.DataArray(np.zeros([len(Adot.channel), len(Adot.wavelength), len(VERTEX_LIST)]),
                           dims = ['channel', 'wavelength', 'vertex'],
                           coords={'channel': Adot.channel.values, 
                                   'wavelength': Adot.wavelength,
                                   'vertex': VERTEX_LIST})

    
    
# load in the data
subj_temp =  f'{subject}/nirs/{subject}_task-{TASK}_run-01_nirs.snirf'
file_name = os.path.join(ROOT_DIR, subj_temp)
print(file_name)
rec = io.read_snirf(file_name)[0]

# replace an NaNs in the data
rec['amp'] = rec['amp'].sel(channel=channel)
rec['amp'] = rec['amp'].where( ~rec['amp'].isnull(), 1e-18 )
rec['amp'] = rec['amp'].where( rec['amp']>0, 1e-18 )

# if first value is 1e-18 then replace with second value
indices = np.where(rec['amp'][:,0,0] == 1e-18)
rec['amp'][indices[0],0,0] = rec['amp'][indices[0],0,1]
indices = np.where(rec['amp'][:,1,0] == 1e-18)
rec['amp'][indices[0],1,0] = rec['amp'][indices[0],1,1]
    
rec['amp'] = rec['amp'].pint.dequantify().pint.quantify('V')

# then we calculate the masks for each metric: SD distance and mean amplitude
rec, chs_pruned = pf.prune_channels(rec)
rec['od'] = nirs.int2od(rec['amp'])
rec['od'] = xr.where(np.isinf(rec['od']), 1e-16, rec['od'])
rec['od'] = xr.where(np.isnan(rec['od']), 1e-16, rec['od'])

for vertex in VERTEX_LIST:
    
    print(f'\tvertex = {vertex}')
    # make the blob of activation 
    blob_img = synthetic_hrf.build_blob_from_seed_vertex(head, vertex = vertex, scale = BLOB_SIGMA)

    # project back to channel space using the forward model
    syn_HRF_chan = synthetic_hrf.hrfs_from_image_reco(blob_img, tbasis, Adot)
    syn_HRF_chan = xr.where(np.isnan(syn_HRF_chan), 1e-16, syn_HRF_chan)
    syn_HRF_chan = xr.where(np.isinf(syn_HRF_chan), 1e-16, syn_HRF_chan)

    # scale the weights by the scale factor
    syn_HRF_chan_scaled = syn_HRF_chan / syn_HRF_chan.sel(wavelength=850).max('time').max('channel').values * SCALE_FACTOR

    # build the stim dataframe
    HRF_stim_df = synthetic_hrf.build_stim_df(num_stims=15, stim_dur=TRANGE_HRF[1], trial_types=['stim'], min_interval=10, max_interval=20)
    while HRF_stim_df['onset'].iloc[-1] + TRANGE_HRF[1] > rec['od'].time[-1]:
        HRF_stim_df = HRF_stim_df[:-1]
    
    # add the HRF to the OD timeseries according to the stim dataframe
    rec['od_wHRF'] = synthetic_hrf.add_hrf_to_od(rec['od'], syn_HRF_chan_scaled, HRF_stim_df)
    rec['od_wHRF'].time.attrs['units'] = units.s
    
    # do TDDR and lowpass filter if the noise model is OLS
    if cfg_GLM['noise_model'] == 'ols':
        rec['od_wHRF'] = motion.tddr(rec['od_wHRF'])
        rec['od_wHRF'] = rec['od_wHRF'].where( ~rec['od_wHRF'].isnull(), 1e-18 ) 

        # low pass filter the data at 0.5 Hz
        rec["od_wHRF"] = freq_filter(rec["od_wHRF"], 
                                    0*units.Hz, 
                                    0.5*units.Hz)

    # convert to concentration
    rec['conc_o'] = nirs.od2conc(rec['od_wHRF'], rec.geo3d, dpf)

    # run the GLM 
    results, hrf_estimate, hrf_mse = pf.GLM([rec['conc_o']], cfg_GLM, rec.geo3d, chs_pruned, [HRF_stim_df])    
    
    # convert the MSE back into OD
    hrf_mse = hrf_mse.transpose('channel', 'time', 'chromo', 'trial_type').pint.dequantify().pint.quantify('molar**2')
    od_mse = xr.dot(E**2, hrf_mse, dim =['chromo']) * 1 * units.mm**2

    # set bad values in mse_t to the bad value threshold
    amp_vals = rec['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
    idx_amp = np.where(amp_vals < cfg_mse['mse_amp_thresh'])[0]
    idx_sat = np.where(chs_pruned == 0.0)[0]
    bad_indices = np.unique(np.concat([idx_amp, idx_sat]))
    
    od_mse.loc[:,channel.isel(channel=bad_indices),:] = cfg_mse['mse_val_for_bad_data']
    od_mse = xr.where(od_mse < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], od_mse)

    # take the mean MSE over the T_WIN
    od_mse_mag = od_mse.sel(time=slice(T_WIN[0], T_WIN[1])).mean('time')

    # save C_meas for vertex and subject 
    C_meas_list.loc[:,:, vertex] = od_mse_mag.squeeze()
    
# save the C_meas as the within subject variance for all subjects  
print('Saving the data')
with open(os.path.join(SAVE_DIR, f"C_meas_sub-{subject}_task-{TASK}_blob-{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_{cfg_GLM['noise_model']}.pkl"), 'wb') as f:
    pickle.dump(C_meas_list, f)

print('Job Complete.')

# %%
