#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submit the batch jobs using this python script.
Ensure the path to the shell script matches your directory structure
Specify the image recon parameters you want to test
Configurables: 
- CODE_DIR: the directory where all the code from the git repository is held 
    
choose the image recon parameters to test 
- alpha_meas_list: select range of alpha measurement
- alpha_spatial_list: select range of alpha spatial 
- sigma_brain_list: select range of sigma brain 
- sigma_scalp_list: select range of sigma scalp 

@author: lcarlton
"""

import subprocess

CODE_DIR = '/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/single_wl_aug'

alpha_meas_list = [10 ** i for i in range(-1, 3)]
alpha_spatial_list = [1e-3, 1e-2]
sigma_brain_list = [0, 1, 3, 5]
sigma_scalp_list = [0, 1, 5, 10, 20]

for sigma_brain in sigma_brain_list:
    
    for sigma_scalp in sigma_scalp_list:
        
        if sigma_brain == 0 and sigma_scalp != 0:
            continue
        elif sigma_brain != 0 and sigma_scalp == 0:
            continue
        
        for alpha_meas in alpha_meas_list:
            
            for alpha_spatial in alpha_spatial_list:
     	               
                qsub_command = f'qsub {CODE_DIR}/batch_shell_script_single_wl_aug.sh {alpha_meas} {alpha_spatial} {sigma_brain} {sigma_scalp}'
                subprocess.run(qsub_command, shell=True)
    
