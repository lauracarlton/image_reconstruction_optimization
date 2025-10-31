#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use this script to generate the images that are included in Figure 5 of 
"Surface-Based Image Reconstruction Optimization for High-Density Functional Near Infrared Spectroscopy"

Configurables:
- ROOT_DIR: path to your bids dataset
- BLOB_SIGMA: the standard deviation of the Gaussian blob of activation (mm)
- TASK: which of the tasks in the BIDS dataset was augmented 
- SCALE_FACTOR: the amplitude of the maximum change in 850nm OD in channel space
- GLM_METHOD: which solving method was used in preprocessing of augmented data - ols or ar_irls
- alpha_spatial_sb: the value of alpha_spatial desired for plotting the metrics using spatial basis functions
- alpha_spatial_nosb: the value of alpha_spatial desired for plotting the metrics using no spatial basis functions

Output: 
- Figure saved showing all the metrics across the range of alpha_meas, sigma_brain and sigma_scalp provided

@author: lcarlton
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

ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids')
BLOB_SIGMA = 15
TASK = 'RS'
SCALE_FACTOR = 0.02
GLM_METHOD = 'ols'  # add if consistent with other figures

alpha_spatial_sb = 1e-2
alpha_spatial_no_sb = 1e-3

#%% LOAD DATA
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
SAVE_PLOT = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'figures')

os.makedirs(SAVE_PLOT, exist_ok=True)

with open(os.path.join(SAVE_DIR, f'COMPILED_METRIC_RESULTS_task-{TASK}_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_single_wl_{GLM_METHOD}.pkl'), 'rb') as f:
    RESULTS = pickle.load(f)

sigma_brain_list = RESULTS['FWHM'].coords['sigma_brain'].values
sigma_scalp_list = RESULTS['FWHM'].coords['sigma_scalp'].values
alpha_meas_list = RESULTS['FWHM'].coords['alpha_meas'].values
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
                alpha_spatial = alpha_spatial_no_sb
                label = 'No Spatial Basis'
            else:
                continue

            ax.errorbar(
                alpha_meas_list,
                mean.sel(sigma_brain=sigma_brain, sigma_scalp=sigma_scalp, alpha_spatial=alpha_spatial),
                se.sel(sigma_brain=sigma_brain, sigma_scalp=sigma_scalp, alpha_spatial=alpha_spatial),
                color=color, lw=lw, ls=l, capsize=capsize, capthick=capthick, label=label
            )

    ax.set_ylabel(y_label)
    ax.set_xlabel('$\\alpha_{meas}$')
    ax.set_xscale('log')
    ax.grid(True)

#%% PLOT SETTINGS
fig = plt.figure(figsize=[60, 30]) 
gs = gridspec.GridSpec(2, 3, wspace=0.25, hspace=0.3)  # All rows have equal height

cmap = sns.color_palette("Spectral", as_cmap=True)
colors = cmap([0.05, 0.25, 0.7, 0.85, 1])
capsize, capthick, lw = 10, 4, 6
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
        .sel(sigma_brain=sigma_brain, sigma_scalp=sigma_scalp_list[-1], alpha_spatial=alpha_spatial_sb)
        .isel(alpha_meas=0),
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
           fontsize=60,
           frameon=False,
           bbox_to_anchor=(0.5, -0.1))  

#% SAVE FIGURE
plt.tight_layout() 
plt.savefig(
    os.path.join(SAVE_PLOT, f'FIG5_{BLOB_SIGMA}mm_assb-{alpha_spatial_sb}_asnosb-{alpha_spatial_no_sb}_metrics_augRS_single_wl_{GLM_METHOD}.png'),
    dpi=300
)
plt.show()

# %%
