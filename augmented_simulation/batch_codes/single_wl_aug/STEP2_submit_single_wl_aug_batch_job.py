#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP2_submit_single_wl_aug_batch_job.py

Batch job submission script for single-wavelength augmented simulations. This
module generates and submits batch jobs to a cluster scheduler (SGE) to
parallelize the computation of image reconstruction metrics across multiple
parameter combinations (alpha_meas, alpha_spatial, sigma_brain, sigma_scalp).

Usage
-----
Edit the CONFIG section (CODE_DIR, alpha_meas_list, etc.) then run::

    python STEP2_submit_single_wl_aug_batch_job.py

Inputs
------
- single_wl_metrics_batch_aug.py: The worker script to be executed by each batch job.
- batch_shell_script_single_wl_aug.sh: Shell script template for SGE batch submission.

Configurables (defaults shown)
-----------------------------
Directory Parameters:
- CODE_DIR (str): '/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/single_wl_aug'
    - Directory containing the batch worker script and shell script.

Image Reconstruction Parameters to Test:
- alpha_meas_list (list[float]): [10 ** i for i in range(-1, 3)]
    - Range of measurement regularization parameters to sweep (0.1, 1, 10, 100).
- alpha_spatial_list (list[float]): [1e-3, 1e-2]
    - Range of spatial regularization parameters to sweep.
- sigma_brain_list (list[int]): [0, 1, 3, 5]
    - Range of brain spatial basis function widths to test (mm).
- sigma_scalp_list (list[int]): [0, 1, 5, 10, 20]
    - Range of scalp spatial basis function widths to test (mm).

Job Submission Logic:
- Skips invalid combinations where sigma_brain=0 and sigma_scalp!=0, or vice versa.
- Submits one job per valid parameter combination using qsub (SGE).

Outputs
-------
- Submits batch jobs to SGE cluster scheduler via qsub command.
- Each job executes single_wl_metrics_batch_aug.py with specific parameter values.
- Number of jobs submitted = len(alpha_meas_list) × len(alpha_spatial_list) × 
  len(valid sigma combinations).
- Job output and error logs are controlled by batch_shell_script_single_wl_aug.sh.

Dependencies
------------
- subprocess

Author: Laura Carlton
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

