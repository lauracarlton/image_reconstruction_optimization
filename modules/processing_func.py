#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:03:03 2025

@author: lcarlton
"""


import cedalion
import cedalion.datasets as datasets
import cedalion.imagereco.forward_model as fw
import cedalion.io as io
import cedalion.models.glm as glm
import cedalion.nirs as nirs
from scipy.spatial.distance import cdist
import cedalion.sigproc.frequency as frequency
from cedalion.sigproc import quality 
import pandas as pd
import xarray as xr
from cedalion import units
import cedalion.sigproc.motion_correct as motion
import numpy as np
import os.path
import pickle
from functools import reduce
import cedalion.xrutils as xrutils 
import operator



def prune_channels(rec, amp_thresh=[1e-3, 0.84]*units.V, sd_thresh=[0, 45]*units.mm, snr_thresh=5):
    
    amp_thresh_sat = [0.*units.V, amp_thresh[1]]
    amp_thresh_low = [amp_thresh[0], 1*units.V]

    _, sd_mask = quality.sd_dist( rec['amp'], rec.geo3d,  sd_thresh)
    _, amp_mask = quality.mean_amp( rec['amp'],  amp_thresh)
    _, amp_mask_sat = quality.mean_amp( rec['amp'],  amp_thresh_sat)
    _, amp_mask_low = quality.mean_amp( rec['amp'],  amp_thresh_low)
    _, snr_mask = quality.snr( rec['amp'],  snr_thresh)

    masks = [sd_mask, amp_mask, snr_mask]

    rec['amp_pruned'], drop_list = quality.prune_ch(rec['amp'], masks, "all", flag_drop=False)
    
    chs_pruned = xr.DataArray(np.zeros(rec['amp'].shape[0]), dims=["channel"], coords={"channel": rec['amp'].channel})

    #i initialize chs_pruned to 0.4
    chs_pruned[:] = 0.4
    chs_pruned[~snr_mask[:,0]] = 0.19
    chs_pruned[~amp_mask_sat[:,0]] = 0
    chs_pruned[~amp_mask_low[:,0]] = 0.8

    return rec, chs_pruned


def prune_mask_ts(ts, pruned_chans):
    '''
    Function to mask pruned channels with NaN .. essentially repruning channels
    Parameters
    ----------
    ts : data array
        time series from rec[rec_str].
    pruned_chans : list or array
        list or array of channels that were pruned prior.

    Returns
    -------
    ts_masked : data array
        time series that has been "repruned" or masked with data for the pruned channels as NaN.

    '''
    mask = np.isin(ts.channel.values, pruned_chans)
    
    if ts.ndim == 3 and ts.shape[0] == len(ts.channel):
        mask_expanded = mask[:, None, None]  # (chan, wav, time)
    elif ts.ndim == 3 and ts.shape[1] == len(ts.channel):
        mask_expanded = mask[None, :, None]  # (chrom, chan, time)
    else:
        raise ValueError("Expected input shape to be either (chan, dim, time) or (dim, chan, time)")

    ts_masked = ts.where(~mask_expanded, np.nan)
    return ts_masked

def median_filter(rec, median_filt = 3):
    pad_width = 1  # Adjust based on the kernel size
    padded_amp = rec['amp'].pad(time=(pad_width, pad_width), mode='edge')
    # Apply the median filter to the padded data
    filtered_padded_amp = padded_amp.rolling(time=median_filt, center=True).reduce(np.median)
    # Trim the padding after applying the filter
    rec['amp'] = filtered_padded_amp.isel(time=slice(pad_width, -pad_width))
    return rec    

#%% GLM FUNCTIONALITY 

def GLM(runs, cfg_GLM, geo3d, pruned_chans_list, stim_list):

    # 1. need to concatenate runs 
    if len(runs) > 1:
        Y_all, stim_df, runs_updated = concatenate_runs(runs, stim_list)
    else:
        Y_all = runs[0]
        stim_df = stim_list[0]
        runs_updated = runs
        
    run_unit = Y_all.pint.units
    # 2. define design matrix
    dms = glm.design_matrix.hrf_regressors(
                                    Y_all,
                                    stim_df,
                                    glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                )


    # Combine drift and short-separation regressors (if any)
    if cfg_GLM['do_drift']:
        drift_regressors = get_drift_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_drift_legendre']:
        drift_regressors = get_drift_legendre_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_short_sep']:
        ss_regressors = get_short_regressors(runs_updated, pruned_chans_list, geo3d, cfg_GLM)
        dms &= reduce(operator.and_, ss_regressors)

    dms.common = dms.common.fillna(0)

    # 3. get betas and covariance
    results = glm.fit(Y_all, dms, noise_model=cfg_GLM['noise_model']) 
    betas = results.sm.params
    cov_params = results.sm.cov_params()

    # 4. estimate HRF and MSE
    basis_hrf = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(Y_all)

    trial_type_list = stim_df['trial_type'].unique()

    hrf_mse_list = []
    hrf_estimate_list = []

    for trial_type in trial_type_list:
        
        betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
        hrf_estimate = estimate_HRF_from_beta(betas_hrf, basis_hrf)
        
        cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                            regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
                                    )
        hrf_mse = estimate_HRF_cov(cov_hrf, basis_hrf)

        hrf_estimate = hrf_estimate.expand_dims({'trial_type': [ trial_type ] })
        hrf_mse = hrf_mse.expand_dims({'trial_type': [ trial_type ] })

        hrf_estimate_list.append(hrf_estimate)
        hrf_mse_list.append(hrf_mse)

    hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
    hrf_estimate = hrf_estimate.pint.quantify(run_unit)

    hrf_mse = xr.concat(hrf_mse_list, dim='trial_type')
    hrf_mse = hrf_mse.pint.quantify(run_unit**2)

    # set universal time so that all hrfs have the same time base 
    fs = frequency.sampling_rate(runs[0]).to('Hz')
    before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = 'second'

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = 'second'

    return results, hrf_estimate, hrf_mse


def estimate_HRF_cov(cov, basis_hrf):

    basis_hrf = basis_hrf.rename({'component':'regressor_c'})
    basis_hrf = basis_hrf.assign_coords(regressor_c=cov.regressor_c.values)

    tmp = xr.dot(cov, basis_hrf, dims='regressor_c')

    tmp = tmp.rename({'regressor_r':'regressor'})
    basis_hrf = basis_hrf.rename({'regressor_c':'regressor'})

    mse_t = xr.dot(basis_hrf, tmp, dims='regressor')

    return mse_t

def estimate_HRF_from_beta(betas, basis_hrf):
        
    basis_hrf = basis_hrf.rename({'component':'regressor'})
    basis_hrf = basis_hrf.assign_coords(regressor=betas.regressor.values)

    hrf_estimate = xr.dot(betas, basis_hrf, dims='regressor')

    hrf_estimates_blcorr = hrf_estimate - hrf_estimate.sel(time = hrf_estimate.time[hrf_estimate.time<0]).mean('time')

    return hrf_estimates_blcorr

def get_drift_regressors(runs, cfg_GLM):
    
    drift_regressors = []
    i=0
    for i, run  in enumerate(runs):
        drift = glm.design_matrix.drift_regressors(run, cfg_GLM['drift_order'])
        drift.common = drift.common.assign_coords({'regressor': [f'Drift {x} run {i}' for x in range(cfg_GLM['drift_order']+1)]})
        drift_regressors.append(drift)
        
    return drift_regressors

def get_drift_legendre_regressors(runs, cfg_GLM):

    drift_regressors = []
    i=0
    for i, run  in enumerate(runs):

        drift = glm.design_matrix.drift_legendre_regressors(run, cfg_GLM['drift_order'])
        drift.common = drift.common.assign_coords({'regressor': [f'Drift {x} run {i}' for x in range(cfg_GLM['drift_order']+1)]})
        drift_regressors.append(drift)

    return drift_regressors

def get_short_regressors(runs, pruned_chans_list, geo3d, cfg_GLM):
    ss_regressors = []
    i=0
    for run, pruned_chans in zip(runs, pruned_chans_list):

        rec_pruned = prune_mask_ts(run, pruned_chans) # !!! how is this affected when using pruned data
        _, ts_short = cedalion.nirs.split_long_short_channels(
                                rec_pruned, geo3d, distance_threshold= cfg_GLM['distance_threshold']  # !!! change to rec_pruned once NaN prob fixed
                                )

        short = glm.design_matrix.average_short_channel_regressor(ts_short)
        short.common = short.common.reset_coords('samples', drop=True)
        short.common = short.common.assign_coords({'regressor': [f'short run {i}']})
        ss_regressors.append(short)
        i = i+1

    return ss_regressors

def concatenate_runs(runs, stim):

    CURRENT_OFFSET = 0
    runs_updated = []
    stim_updated = []

    for s, ts in zip(stim, runs):
        time = ts.time.values
        new_time = time + CURRENT_OFFSET

        ts_new = ts.copy(deep=True)
        ts_new = ts_new.pint.to('molar')
        ts_new = ts_new.assign_coords(time=new_time)

        stim_shift = s.copy()
        stim_shift['onset'] += CURRENT_OFFSET

        stim_updated.append(stim_shift)
        runs_updated.append(ts_new)

        CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])

    Y_all = xr.concat(runs_updated, dim='time')
    Y_all.time.attrs['units'] = units.s
    stim_df = pd.concat(stim_updated, ignore_index = True)

    return Y_all, stim_df, runs_updated

#%% SPATIAL SMOOTHING 

def get_spatial_smoothing_kernel(V_ras, sigma_mm=80*units.mm):

    # distance matrix. float32 cuts memory in half vs float64
    D = cdist(V_ras.astype(np.float32), V_ras.astype(np.float32))  # shape (N, N), distances in mm

    # memory friendly in-place operations for calculating the gausisian weights
    W = D.astype(np.float32, copy=True)   # reuse same shape
    np.square(W, out=W)                   # W = D^2
    W /= (sigma_mm ** 2)                  # W = D^2 / sigma^2
    np.negative(W, out=W)                 # W = - D^2 / sigma^2
    np.exp(W, out=W)                      # W = exp(...)
    # set all columns of W to zero where log10(Adot_sum) < -2
    # set all rows of W to zero where log10(Adot_sum) < -2
    # set the diagonal to 1 in the rows where log10(Adot_sum) < -2
    W[W<1e-3] = 0

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums

    return W
    
def compute_Hglobal_from_PCA(data_xr, variances, W):
    """
    data_xr: xarray.DataArray with dims ('trial_type','subj','chromo','vertex','reltime' or 'time')
    W: (P x P) Gaussian smoothing operator over vertices (same vertex indexing as data_xr)
    returns: xarray.DataArray H_global with same shape/coords/units as data_xr
    """
    # figure out time dim name
    time_dim = 'reltime' if 'reltime' in data_xr.dims else 'time'

    # drop units for speed / to avoid pint assignment issues
    has_units = hasattr(data_xr, 'pint') and getattr(data_xr.pint, 'units', None) is not None
    units = data_xr.pint.units if has_units else None
    dx = data_xr.pint.dequantify() if has_units else data_xr

    # output container (unitless for now)
    Hglob = dx.copy(deep=True)

    # precompute W^T once
    # WT = W.T.astype(np.float32, copy=False)
    W_orig = W.copy()
    # build label->index maps for fast .data assignment


    tt_idx = {v: i for i, v in enumerate(dx.trial_type.values)} if "trial_type" in dx.dims else None
    sj_idx = {v:i for i,v in enumerate(dx.subj.values)}
    ch_idx = {v:i for i,v in enumerate(dx.chromo.values)}

    trial_types = dx.trial_type.values if "trial_type" in dx.dims else [None]

    # iterate slices
    for tt in trial_types:
        for sj in dx.subj.values:
            for ch in dx.chromo.values:
                # H slice as (T x P)

                sel_dict = dict(subj=sj, chromo=ch)
                if tt is not None:
                    sel_dict["trial_type"] = tt

                H = dx.sel(sel_dict).transpose(time_dim, 'vertex').values.astype(np.float32, copy=False)
                var = variances.sel(sel_dict).values

                W = W_orig # / var
                row_sums = W.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                W /= row_sums

                WT = W.T.astype(np.float32, copy=False)

                mean = H.mean(axis=1, keepdims=True)
                H_centered = H - mean
                
                # economy SVD: H = U @ diag(S) @ Vt
                U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)   # U:(T,r), S:(r,), Vt:(r,P)
                # smooth spatial PCs: Vt* = Vt @ W^T  (since V* = W @ V, and (V*)^T = V^T W^T)
                Vt_star = Vt @ WT                                   # (r,P)              
                Hglob_slice = U @ np.diag(S) @ Vt_star  # (use broadcasting for diag(S))

                # 4) reconstruct the smoothed-global in vertex space and add back mean
                Hglob_slice_recon = Hglob_slice + mean

                # Hglob_slice = (U * S) @ Vt_star                     # (T,P)
                
                # write back as (vertex x time)
                if tt is not None:
                    Hglob.data[tt_idx[tt], sj_idx[sj], ch_idx[ch], :, :] = Hglob_slice_recon.T
                else:
                    Hglob.data[sj_idx[sj], ch_idx[ch], :, :] = Hglob_slice_recon.T
    # reattach units if present
    if has_units:
        Hglob = Hglob.pint.quantify(units)

    final_dims = ['trial_type'] if 'trial_type' in dx.dims else []
    final_dims += ['subj', 'chromo', 'vertex', time_dim]

    # return with original dim order
    return Hglob.transpose(*final_dims)
