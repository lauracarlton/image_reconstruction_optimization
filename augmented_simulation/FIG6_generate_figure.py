#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG6_generate_figure.py

Figure generation for dual-wavelength augmented simulation results. This
module loads the dual-wavelength image reconstruction metrics computed in
FIG6_STEP2 and creates publication-quality visualizations comparing HbO and
HbR reconstruction quality across regularization parameters for both direct
and indirect reconstruction methods.

Usage
-----
Edit the CONFIG section (ROOT_DIR, BLOB_SIGMA, TASK, etc.) then run::

    python FIG6_generate_figure.py

Inputs
------
- Pickle file from STEP3_compile_results_dual_wl.py located at
  <ROOT_DIR>/derivatives/cedalion/augmented_data/ with filename:
  COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_lR-{lambda_R}_{NOISE_MODEL}_dual_wl.pkl
  containing dual-wavelength reconstruction metrics for various alpha_meas,
  alpha_spatial, sigma_brain, sigma_scalp values, and seed vertex configurations.

Configurables (defaults shown)
-----------------------------
Data Storage Parameters:
- ROOT_DIR (str): '/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids'
    - Root directory containing simulation results.

Augmentation Parameters (must match STEP2):
- BLOB_SIGMA (float): 15
    - Standard deviation of Gaussian activation blob in mm.
- TASK (str): 'RS'
    - Task identifier matching the augmented dataset.
- SCALE_FACTOR (float): 0.02
    - Amplitude of the maximum change in 850nm OD in channel space.
- NOISE_MODEL (str): 'ar_irls'
    - GLM solving method used in preprocessing of augmented data (ols or ar_irls).


Plotting Parameters:
- alpha_spatial_sb (float): 1e-2
    - Value of alpha_spatial for plotting metrics using spatial basis functions.
- alpha_spatial_nosb (float): 1e-3
    - Value of alpha_spatial for plotting metrics without spatial basis functions.
- sigma_brain (int): 1
    - Value of sigma_brain (mm) to use for spatial basis function plotting.
- sigma_scalp (int): 5
    - Value of sigma_scalp (mm) to use for spatial basis function plotting.
- lambda_R (float): 1e-6
    - scaling parameter for the image prior used in reconstruction.
    
Outputs
-------
- Publication-ready figure (Figure 6) saved to <ROOT_DIR>/derivatives/cedalion/figures/
  with filename: FIG6_task-{TASK}_blob-{BLOB_SIGMA}mm_assb-{alpha_spatial_sb}_asnosb-{alpha_spatial_nosb}_lR-{lambda_R}_{NOISE_MODEL}_metrics_dual_wl.png
  showing 7 panels:
  - FWHM HbO vs alpha_meas
  - CNR HbO vs alpha_meas
  - Contrast ratio HbO vs alpha_meas
  - Localization error HbO vs alpha_meas
  - Brain→Scalp crosstalk HbO vs alpha_meas (log scale)
  - HbO→HbR crosstalk vs alpha_meas
  - HbR→HbO crosstalk vs alpha_meas
  Each panel shows direct method (solid line) and indirect method (dashed line)
  for both spatial basis and no-spatial-basis conditions.

Dependencies
------------
- numpy, matplotlib, seaborn, pickle

Author: Laura Carlton
"""

#%% IMPORTS
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams['font.size'] = 80

#%% CONFIGURABLES
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids_v2')

BLOB_SIGMA = 15
SCALE_FACTOR = 0.02
NOISE_MODEL = 'ar_irls'
TASK = 'RS'
vline_val = 1e4

alpha_spatial_sb = 1e-2
alpha_spatial_nosb = 1e-3
sigma_brain = 1
sigma_scalp = 5
lambda_R = 1e-6
#%% LOAD DATA
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
SAVE_PLOT = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'figures')

os.makedirs(SAVE_PLOT, exist_ok=True)

with open(os.path.join(SAVE_DIR, f'COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_lR-{lambda_R}_{NOISE_MODEL}_dual_wl.pkl'), 'rb') as f:
    RESULTS = pickle.load(f)

# alpha_meas_list = RESULTS['FWHM'].coords['alpha_meas'].values
alpha_meas_list = RESULTS['FWHM_HbO_direct'].coords['alpha_meas'].values[3:]
VERTEX_LIST = RESULTS['FWHM_HbO_direct'].vertex.values
RESULTS['FWHM_HbO_direct'] = RESULTS['FWHM_HbO_direct'] 
RESULTS['FWHM_HbO_indirect'] = RESULTS['FWHM_HbO_indirect'] 


#%% HELPER FUNCTIONS
def mean_se(data):
    """Compute mean and SE over vertices."""
    mean = data.mean('vertex')
    se = data.std('vertex') / np.sqrt(len(VERTEX_LIST))
    return mean, se


def plot_metric(ax, metric_name, y_label, colors):
    """Plot direct/indirect curves for both SB and no-SB conditions."""
    mean_dir, se_dir = mean_se(RESULTS[f'{metric_name}_direct'])
    mean_ind, se_ind = mean_se(RESULTS[f'{metric_name}_indirect'])

    # Define condition sets
    conditions = [
        dict(sigma_brain=sigma_brain, sigma_scalp=sigma_scalp, alpha_spatial=alpha_spatial_sb, alpha_meas=alpha_meas_list,
             color=colors[1], label=f'$\\sigma_{{brain}}$ = {sigma_brain} mm; $\\sigma_{{scalp}}$ = {sigma_scalp} mm'),
        dict(sigma_brain=0, sigma_scalp=0, alpha_spatial=alpha_spatial_nosb, alpha_meas=alpha_meas_list,
             color=colors[0], label='No Spatial Basis')
    ]

    for cond in conditions:
        label = cond.pop("label")
        color = cond.pop("color")

        # Direct
        ax.errorbar(alpha_meas_list*0.7,
                    mean_dir.sel(**cond),
                    se_dir.sel(**cond),
                    color=color, lw=lw, ls='-', capsize=capsize, capthick=capthick,
                    label=label)

        # Indirect 
        ax.errorbar(alpha_meas_list*1.3,
                    mean_ind.sel(**cond),
                    se_ind.sel(**cond),
                    color=color, lw=lw, ls='--', capsize=capsize, capthick=capthick,
                    )

    ax.set_ylabel(y_label)
    ax.set_xlabel('$\\alpha_{meas}$')
    ax.set_xscale('log')
    ax.grid(True)


#%% COLORS & PLOT SETTINGS
fig = plt.figure(figsize=(90, 40))
gs = gridspec.GridSpec(2, 4)

cmap = sns.color_palette("Spectral", as_cmap=True)
colors = ('red', 'purple', 'green', 'blue')
capsize, capthick, lw = 30, 8, 10

#% DEFINE METRICS AND SUBPLOTS
metrics = [
    ('FWHM_HbO', 'FWHM HbO (mm)'),
    ('CNR_HbO', 'CNR HbO'),
    ('contrast_ratio_HbO', 'Contrast Ratio HbO'),
    ('localization_error_HbO', 'Localization Error HbO (mm)'),
    ('crosstalk_brainVscalp_HbO', 'Crosstalk: Brain→Scalp HbO'),
    ('crosstalk_HbOVHbR', 'Crosstalk: HbO→HbR'),
    ('crosstalk_HbRVHbO', 'Crosstalk: HbR→HbO'),
]
mean_dir, _ = mean_se(RESULTS['FWHM_HbO_direct'])
mean_ind, _ = mean_se(RESULTS['FWHM_HbO_indirect'])

axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(len(metrics))]
axes[0].plot(alpha_meas_list[0], mean_dir[0].isel(sigma_brain=0, sigma_scalp=0, alpha_spatial=0), '-k', label='Direct', lw=lw-1)
axes[0].plot(alpha_meas_list[0], mean_ind[0].isel(sigma_brain=0, sigma_scalp=0, alpha_spatial=0), '--k', label='Indirect', lw=lw-1)

#% PLOT ALL METRICS
for (metric_name, y_label), ax in zip(metrics, axes):
    ax.axvline(vline_val, color='k', linestyle='--', lw=30)
    plot_metric(ax, metric_name, y_label, colors)

# Apply log y-scale to specific plot if desired
axes[4].set_yscale('log')  # Crosstalk brain→scalp
axes[1].set_yscale('log')  # CNR

#% LEGEND
legend_ax = fig.add_subplot(gs[1, 3])
legend_ax.axis('off')
handles, labels = axes[0].get_legend_handles_labels()
legend_ax.legend(handles, labels, loc='center', ncol=1, fontsize=100)

plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_PLOT, f'FIG6_task-{TASK}_blob-{BLOB_SIGMA}mm_assb-{alpha_spatial_sb}_asnosb-{alpha_spatial_nosb}_lR-{float(lambda_R)}_{NOISE_MODEL}_metrics_dual_wl.png'), 
    dpi=200
)
plt.show()

# %%
