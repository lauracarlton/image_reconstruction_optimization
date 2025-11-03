#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_results.py

Collect per-subject HRF estimates produced by FIG8_STEP1 and combine them
into a single per-subject archive for downstream group-level analysis.

This small utility scans a BIDS-like dataset folder for subject subfolders,
loads each subject's gzipped pickle produced by the HRF estimation step
(expected keys: 'hrf_per_subj', 'hrf_mse_per_subj', 'bad_indices'), and
concatenates the HRF and MSE xarray objects along a new 'subj' dimension.
The combined results are written as a gzipped pickle under
`<ROOT_DIR>/derivatives/processed_data/`.

Usage
-----
Edit the CONFIG section variables below (ROOT_DIR, NOISE_MODEL, REC_STR,
and EXCLUDED) to match your dataset, then run from a Python interpreter:

    python compile_results.py

Configurables (defaults shown in script)
---------------------------------------
- ROOT_DIR: path to the dataset root containing subject folders.
- NOISE_MODEL: label used when naming per-subject HRF output files.
- REC_STR: record string used in per-subject filenames ('conc_o' by default).
- EXCLUDED: list of subject IDs to skip (e.g. problematic subjects).

Outputs
-------
Writes a single gzipped pickle to:
  <ROOT_DIR>/derivatives/processed_data/{REC_STR}_hrf_estimates_per_subj_{NOISE_MODEL}.pkl.gz
Containing a dict with keys:
- 'hrf_per_subj' : xarray concatenation along 'subj'
- 'mse_per_subj' : xarray concatenation along 'subj'
- 'bad_indices'   : list of per-subject bad-indices arrays

Dependencies
------------
Requires Python packages: xarray, gzip, pickle, and that per-subject
HRF files exist at the expected paths.

Author: Laura Carlton
"""

import os
import pickle
import gzip
import xarray as xr

ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids')
NOISE_MODEL = 'ar_irls'
REC_STR = 'conc_o'
EXCLUDED = ['sub-538', 'sub-549', 'sub-547']

dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if 'sub' in d and d not in EXCLUDED]

hrf_all_subj = []
mse_all_subj = []
bad_indices_all_subj = []

for subject in subject_list:

    file_path_pkl = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'processed_data', subject,
                                 f"{subject}_{REC_STR}_hrf_estimates_{NOISE_MODEL}.pkl.gz")
    
    with gzip.open(file_path_pkl, 'rb') as f:
        results = pickle.load(f)

    hrf_all_subj.append(results['hrf_per_subj'].assign_coords(subj=[subject]))
    mse_all_subj.append(results['hrf_mse_per_subj'].assign_coords(subj=[subject]))
    bad_indices_all_subj.append(results['bad_indices'])

hrf_all_subj = xr.concat(hrf_all_subj, dim='subj')
mse_all_subj = xr.concat(mse_all_subj, dim='subj')

all_results = {'hrf_per_subj': hrf_all_subj,
               'mse_per_subj': mse_all_subj,
               'bad_indices': bad_indices_all_subj
               }

with gzip.open(os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'processed_data', f'{REC_STR}_hrf_estimates_per_subj_{NOISE_MODEL}.pkl.gz'), 'wb') as f:
    pickle.dump(all_results, f)

# %%
