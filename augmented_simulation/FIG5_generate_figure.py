#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG5_generate_figure.py

Figure generation for single-wavelength augmented simulation results. This
module loads the image reconstruction metrics computed in FIG5_STEP2 and
creates publication-quality visualizations showing FWHM, CNR, contrast ratio,
localization error, brain-scalp crosstalk, and percentage reconstructed in
brain as functions of regularization parameter alpha_meas.

Usage
-----
Edit the CONFIG section (ROOT_DIR, BLOB_SIGMA, TASK, etc.) then run::

    python FIG5_generate_figure.py

Inputs
------
- Gzipped pickle file from FIG5_STEP2_get_single_wavelength_image_metrics.py
  located at <ROOT_DIR>/derivatives/cedalion/augmented_data/ with filename:
  COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_single_wl_{GLM_METHOD}.pkl
  containing reconstruction metrics for various alpha_meas, alpha_spatial,
  sigma_brain, sigma_scalp values, and seed vertex configurations.

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
- NOISE_MODEL (str): 'ols'
    - GLM solving method used in preprocessing of augmented data (ols or ar_irls).

Plotting Parameters:
- alpha_spatial_sb (float): 1e-2
    - Value of alpha_spatial for plotting metrics using spatial basis functions.
- alpha_spatial_no_sb (float): 1e-3
    - Value of alpha_spatial for plotting metrics without spatial basis functions.

Outputs
-------
- Publication-ready figure (Figure 5) saved to <ROOT_DIR>/derivatives/cedalion/figures/
  with filename: FIG5_{BLOB_SIGMA}mm_assb-{alpha_spatial_sb}_asnosb-{alpha_spatial_no_sb}_metrics_augRS_single_wl_{GLM_METHOD}.png
  showing 6 panels:
  - FWHM vs alpha_meas
  - CNR vs alpha_meas (log scale)
  - Contrast ratio vs alpha_meas
  - Localization error vs alpha_meas
  - Brain→Scalp crosstalk vs alpha_meas (log scale)
  - % predicted by brain vs alpha_meas
  Each panel compares spatial basis function configurations (sigma_brain, sigma_scalp)
  and no-spatial-basis condition across the range of alpha_meas values.

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

#%% SET UP CONFIGURABLES

ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids_v2')
BLOB_SIGMA = 15
TASK = 'RS'
SCALE_FACTOR = 0.02
NOISE_MODEL = 'ar_irls'  # add if consistent with other figures

alpha_spatial_sb = 1e-2
alpha_spatial_nosb = 1e-3

#%% LOAD DATA
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
SAVE_PLOT = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'figures')

os.makedirs(SAVE_PLOT, exist_ok=True)

with open(os.path.join(SAVE_DIR, f'COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_{NOISE_MODEL}_single_wl.pkl'), 'rb') as f:
    RESULTS = pickle.load(f)

sigma_brain_list = RESULTS['FWHM'].coords['sigma_brain'].values
# sigma_scalp_list = RESULTS['FWHM'].coords['sigma_scalp'].values
sigma_scalp_list = [0, 1, 5, 10]
# alpha_meas_list = RESULTS['FWHM'].coords['alpha_meas'].values
alpha_meas_list = RESULTS['FWHM'].coords['alpha_meas'].values[3:]
VERTEX_LIST = RESULTS['FWHM'].vertex.values

#%% HELPER FUNCTIONS
def mean_se(data):
    """Compute mean and SE over vertices."""
    mean = data.mean('vertex')
    se = data.std('vertex') / np.sqrt(len(VERTEX_LIST))
    return mean, se


def plot_metric(ax, metric_name, y_label, colors, ls_list):
    """Plot metric curves for SB and no-SB models."""
    mean, se = mean_se(RESULTS[metric_name])

    for l, sigma_brain in zip(ls_list, sigma_brain_list):
        for jj, sigma_scalp in enumerate(sigma_scalp_list):
            color = colors[jj]

            # Determine SB condition and label
            if sigma_brain > 0 and sigma_scalp > 0:
                alpha_spatial = alpha_spatial_sb
                label = f'$\\sigma_{{scalp}}$ = {sigma_scalp} mm' if sigma_brain == sigma_brain_list[1] else None
            elif sigma_brain == 0 and sigma_scalp == 0:
                alpha_spatial = alpha_spatial_nosb
                label = 'No Spatial Basis'
            else:
                continue

            ax.errorbar(
                alpha_meas_list,
                mean.sel(sigma_brain=sigma_brain, sigma_scalp=sigma_scalp, alpha_spatial=alpha_spatial, alpha_meas=alpha_meas_list),
                se.sel(sigma_brain=sigma_brain, sigma_scalp=sigma_scalp, alpha_spatial=alpha_spatial, alpha_meas=alpha_meas_list),
                color=color, lw=lw, ls=l, capsize=capsize, capthick=capthick, label=label
            )

    ax.set_ylabel(y_label)
    ax.set_xlabel('$\\alpha_{meas}$')
    ax.set_xscale('log')
    ax.grid(True)

#%% PLOT SETTINGS
fig = plt.figure(figsize=[80, 50]) 
gs = gridspec.GridSpec(2, 3, wspace=0.25, hspace=0.3)  # All rows have equal height

cmap = sns.color_palette("Spectral", as_cmap=True)
colors = cmap([0.05, 0.25, 0.7, 1])
colors = ('red', 'purple', 'green', 'blue')

capsize, capthick, lw = 20, 8, 8
ls_list = ['-', '-', '--', ':']

#% DEFINE METRICS AND AXES
metrics = [
    ('FWHM', 'FWHM (mm)'),
    ('CNR', 'CNR'),
    ('contrast_ratio', 'Contrast Ratio'),
    ('localization_error', 'Localization Error (mm)'),
    ('crosstalk_brainVscalp', 'Crosstalk: Brain→Scalp'),
    ('perc_recon_brain', '% predicted by brain')
]

axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(len(metrics))]

#% PLOT ALL METRICS
for (metric_name, y_label), ax in zip(metrics, axes):
    plot_metric(ax, metric_name, y_label, colors, ls_list)

# Apply log y-scale to selected plots
axes[1].set_yscale('log')  # CNR
axes[4].set_yscale('log')  # Crosstalk brain→scalp

#% ADD SECOND LEGEND (brain sigmas)
for ii, sigma_brain in enumerate(sigma_brain_list):
    if sigma_brain == 0:
        continue
    axes[0].plot(
        alpha_meas_list[0],
        RESULTS['FWHM']
        .mean('vertex')
        .sel(sigma_brain=sigma_brain, sigma_scalp=sigma_scalp_list[-1], 
            alpha_spatial=alpha_spatial_sb, alpha_meas=alpha_meas_list[0]),
        'k', ls=ls_list[ii], lw=7,
        label=f'$\\sigma_{{brain}}$ = {sigma_brain} mm'
    )

#% LEGEND
handles, labels = axes[0].get_legend_handles_labels()
plt.subplots_adjust(bottom=0.28, top=0.95, left=0.03, right=0.95, hspace=0.4, wspace=0.5)

# Add a centered legend below all plots
fig.legend(handles, labels,
           loc='lower center',
           ncol=3,
        #    fontsize=60,
           frameon=False,
           bbox_to_anchor=(0.5, -0.01))  

#% SAVE FIGURE
plt.tight_layout() 
plt.savefig(
    os.path.join(SAVE_PLOT, f'FIG5_task-{TASK}_blob-{BLOB_SIGMA}mm_assb-{alpha_spatial_sb}_asnosb-{alpha_spatial_nosb}_{NOISE_MODEL}_metrics_single_wl.png'),
    dpi=200
)
plt.show()

# %%
