
#%%
import os 
import pickle
import gzip
import xarray as xr

noise_model = 'ar_irls'
REC_STR = 'conc_o'

root_dir = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids/"

dirs = os.listdir(root_dir)

excluded = ['sub-538', 'sub-549', 'sub-547'] 


subject_list = [d for d in dirs if 'sub' in d and d not in excluded]

hrf_all_subj = []
mse_all_subj = []
bad_indices_all_subj = []

for subject in subject_list:

    file_path_pkl = os.path.join(root_dir, 'derivatives', 'processed_data', subject,
                                f"{subject}_{REC_STR}_hrf_estimates_{noise_model}.pkl.gz")
    
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

with gzip.open( os.path.join(root_dir, 'derivatives', 'processed_data', f'{REC_STR}_hrf_estimates_per_subj_{noise_model}.pkl.gz'), 'wb') as f:
    pickle.dump(all_results, f)

# %%
