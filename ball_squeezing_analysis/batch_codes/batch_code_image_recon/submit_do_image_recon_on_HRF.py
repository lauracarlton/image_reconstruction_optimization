#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os 
import subprocess

root_dir = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids/"
dirs = os.listdir(root_dir)

excluded = ['sub-538', 'sub-549', 'sub-547'] 

subject_list = [d for d in dirs if 'sub' in d and d not in excluded]

for subj in subject_list:
    
    qsub_command = f'qsub /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/ball_squeezing_analysis/batch_codes/batch_code_image_recon/shell_do_image_recon_on_HRF.sh {subj}'
    subprocess.run(qsub_command, shell=True)

