
#%%
import subprocess
import os 

CODE_DIR = '/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/augmentation'
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids_v2')
EXCLUDED = ['sub-577'] # does not contain RS data 

dirs = os.listdir(ROOT_DIR)
SUBJECT_LIST = [d for d in dirs if 'sub' in d and d not in EXCLUDED]
            
for subject in SUBJECT_LIST:
            
    qsub_command = f'qsub {CODE_DIR}/batch_shell_script_augment_data.sh {subject}'
    subprocess.run(qsub_command, shell=True)

