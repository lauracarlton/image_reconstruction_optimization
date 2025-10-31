#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use this script to generate C_meas that is used downstream to get image metrics used in 
"Surface-Based Image Reconstruction Optimization for High-Density Functional Near Infrared Spectroscopy"

In this script: 
- load in a dataset
- for each run of each subject:
    - basic preprocessing 
        - identify noisy channels
        - convert intensity to OD
        - add synthetic HRF to timeseries 
        - optional -> motion correction using TDDR
        - optional -> lowpass filter at 0.5 Hz
        - convert OD to concentration 
- for each subject
    - concatenate runs and perform HRF estimation 
        - use AR-IRLS or OLS
        - 3rd order drift correction 
        - physiological regression using mean of the 19 mm channels
- save the covariance matrix output 

configure the following parameters:
    - ROOT_DIR: should point to a BIDs data folder
    - HEAD_MODEL: which atlas to use - options in cedalion are Colin27 and ICBM152
    - GLM_METHOD: which solving method was used in preprocessing of augmented data - ols or ar_irls
    - TASK: which of the tasks in the BIDS dataset was augmented 
    - BLOB_SIGMA: the standard deviation of the Gaussian blob of activation (mm) * MUST HAVE UNITS *
    - SCALE_FACTOR: the amplitude of the maximum change in 850nm OD in channel space
    - VERTEX_LIST: list of seed vertices to be used 
    - exclude_subj: any subjects IDs within the BIDs dataset to be excluded
    - TRANGE_HRF: temporal window used to define the HRF
    - STIM_DUR: length of the boxcar function that is convolved with the HRF 
    - T_WIN: window over which to average the peak of the HRF
    - PARAMS_BASIS: parameters used for tau and sigma to define the canonical HRF using a modified gamma function for both HbO and HbR
    - SNR_THRESH: threshold for defining channels with bad SNR
    - DRANGE: threshold for defining low amplitude and saturated channels
    

@author: lcarlton
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

#%% CONFIG 
# DATA STORAGE PARAMS
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids')
exclude_subj = ['sub-577']
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
GLM_METHOD = 'ols' # can select ols for ordinary least squares or ar_irls for autoregressive model
# channel flagging params
D_RANGE = [1e-3, 0.84] # mean signal amplitude min and max
SNR_THRESH = 5 # signal to noise ratio threshold

#%% SETUP DOWNSTREAM CONFIGURABLES
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
os.makedirs(SAVE_DIR, exist_ok=True)

PROBE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'fw', HEAD_MODEL)

dirs = os.listdir(ROOT_DIR)
SUBJECT_LIST = [d for d in dirs if 'sub' in d and d not in exclude_subj]

cmap = plt.cm.get_cmap("jet", 256)

if GLM_METHOD == 'ols':
    DO_TDDR = True
    DO_DRIFT = True
    DO_DRIFT_LEGENDRE = False
    DRIFT_ORDER = 3
elif GLM_METHOD == 'ar_irls':
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
    'noise_model' : GLM_METHOD,
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
subj_temp =  f'{SUBJECT_LIST[0]}/nirs/{SUBJECT_LIST[0]}_task-{TASK}_run-01_nirs.snirf'
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
C_meas_list = xr.DataArray(np.zeros([len(Adot.channel), len(Adot.wavelength), len(SUBJECT_LIST), len(VERTEX_LIST)]),
                           dims = ['channel', 'wavelength', 'subject', 'vertex'],
                           coords={'channel': Adot.channel.values, 
                                   'wavelength': Adot.wavelength,
                                   'subject': SUBJECT_LIST, 
                                   'vertex': VERTEX_LIST})

for ss, subject in enumerate(SUBJECT_LIST):
    
    print(f'subject: {ss+1}/{len(SUBJECT_LIST)}')
    
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
        if cfg_GLM['GLM_METHOD'] == 'ols':
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
        hrf_mse = hrf_mse.transpose('channel', 'time', 'chromo', 'trial_type').pint.to('molar**2')
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
        C_meas_list.loc[:,:, subject, vertex] = od_mse_mag.squeeze()
        
# save the C_meas as the within subject variance for all subjects  
print('Saving the data')
with open(os.path.join(SAVE_DIR, f"C_meas_subj_task-{TASK}_blob-{BLOB_SIGMA.magnitude}mm_scale-{SCALE_FACTOR}_{cfg_GLM['GLM_METHOD']}.pkl"), 'wb') as f:
    pickle.dump(C_meas_list, f)

print('Complete.')

# %%
