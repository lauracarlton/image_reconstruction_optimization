
#%%
import pickle 
import os 
import xarray as xr
import numpy as np 

#%%
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids_v2')
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')

EXCLUDED = ['sub-577'] # does not contain RS data 
BLOB_SIGMA = 15
SCALE_FACTOR = 0.02
TASK = "RS"
NOISE_MODEL = 'ar_irls'

dirs = os.listdir(ROOT_DIR)
SUBJECT_LIST = [d for d in dirs if 'sub' in d and d not in EXCLUDED]
C_meas_all = []


for subject in SUBJECT_LIST: 


    with open(os.path.join(SAVE_DIR, f"C_meas_sub-{subject}_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_{NOISE_MODEL}.pkl"), 'rb') as f:
        C_meas_lst = pickle.load(f)
    
    C_meas_all.append(C_meas_lst)

C_meas_all_xr = xr.concat(C_meas_all, dim='subject')
C_meas_all_xr = C_meas_all_xr.assign_coords({'subject': SUBJECT_LIST})

with open(os.path.join(SAVE_DIR, f"C_meas_subj_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_{NOISE_MODEL}.pkl"), 'wb') as f:
        pickle.dump(C_meas_all_xr, f)


# %%
