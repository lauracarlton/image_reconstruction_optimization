#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use this script to generate the images that are included in Figure 6 of 
"Surface-Based Image Reconstruction Optimization for High-Density Functional Near Infrared Spectroscopy"

Configurables:
- ROOT_DIR -> path to your bids dataset
- BLOB_SIGMA -> size of the Gaussian blob used to generate the augmented data
- SCALE_FACTOR -> scale of the HRF used to generate the augmented data
- NOISE_MODEL -> glm solve method used during data preprocessing
- alpha_spatial_sb -> the value of alpha_spatial desired for plotting the metrics using spatial basis functions
- alpha_spatial_nosb -> the value of alpha_spatial desired for plotting the metrics using no spatial basis functions
- sigma_brain -> value of sigma brain to use for plotting 
- sigma_scalp -> value of sigma scalp to use for plottin 

Output: 
- Figure saved showing all the metrics for both the direct and indirect methods and with and without spatial bses methods across the range of alpha_meas provided

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

#%% CONFIGURABLES
ROOT_DIR = os.path.join('/projectnb', 'nphfnirs', 's', 'datasets', 'BSMW_Laura_Miray_2025', 'BS_bids')

BLOB_SIGMA = 15
SCALE_FACTOR = 0.02
NOISE_MDOEL = 'ols'

alpha_spatial_sb = 1e-2
alpha_spatial_nosb = 1e-3
sigma_brain = 1
sigma_scalp = 5

#%% LOAD DATA
SAVE_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'augmented_data')
SAVE_PLOT = os.path.join(ROOT_DIR, 'derivatives', 'cedalion', 'figures')

os.makedirs(SAVE_PLOT, exist_ok=True)

with open(os.path.join(SAVE_DIR, f'COMPILED_METRIC_RESULTS_blob-{BLOB_SIGMA}mm_scale-{SCALE_FACTOR}_dual_wl_{NOISE_MODEL}.pkl'), 'rb') as f:
    RESULTS = pickle.load(f)

alpha_meas_list = RESULTS['FWHM_HbO_direct'].alpha_meas.values
VERTEX_LIST = RESULTS['FWHM_HbO_direct'].vertex.values

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
        dict(sigma_brain=sigma_brain, sigma_scalp=sigma_scalp, alpha_spatial=alpha_spatial_sb,
             color=colors[2], label=f'$\\sigma_{{brain}}$={sigma_brain} mm; $\\sigma_{{scalp}}$={sigma_scalp} mm'),
        dict(sigma_brain=0, sigma_scalp=0, alpha_spatial=alpha_spatial_nosb,
             color=colors[0], label='No Spatial Basis')
    ]

    for cond in conditions:
        label = cond.pop("label")
        color = cond.pop("color")

        # Direct
        ax.errorbar(alpha_meas_list,
                    mean_dir.sel(**cond),
                    se_dir.sel(**cond),
                    color=color, lw=lw, capsize=capsize, capthick=capthick,
                    label=label)

        # Indirect (dashed, no label)
        ax.errorbar(alpha_meas_list,
                    mean_ind.sel(**cond),
                    se_ind.sel(**cond),
                    color=color, lw=lw, ls='--', capsize=capsize, capthick=capthick)

    ax.set_ylabel(y_label)
    ax.set_xlabel('$\\alpha_{meas}$')
    ax.set_xscale('log')
    ax.grid(True)


#%% COLORS & PLOT SETTINGS
fig = plt.figure(figsize=(80, 40))
gs = gridspec.GridSpec(2, 4)

cmap = sns.color_palette("Spectral", as_cmap=True)
colors = cmap([0.05, 0.25, 0.7, 0.85, 1])
capsize, capthick, lw = 10, 4, 10

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
    plot_metric(ax, metric_name, y_label, colors)

# Apply log y-scale to specific plot if desired
axes[4].set_yscale('log')  # Crosstalk brain→scalp

#% LEGEND
legend_ax = fig.add_subplot(gs[1, 3])
legend_ax.axis('off')
handles, labels = axes[0].get_legend_handles_labels()
legend_ax.legend(handles, labels, loc='center', ncol=1)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PLOT, f'FIG6_augRS_dual_wl_assb-{alpha_spatial_sb}_asnosb-{alpha_spatial_nosb}_{NOISE_MODEL}.png'), dpi=300)
plt.show()

# %%
