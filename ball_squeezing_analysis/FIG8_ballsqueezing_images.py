#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG8_ballsqueezing_images.py

Render surface images (brain and scalp) from image-space reconstructions
produced by the FIG8 pipeline. Loads per-configuration pickles produced by
the image-reconstruction step, applies optional spatial smoothing and
sensitivity masking, and renders/saves screenshots for a chosen set of
trial conditions and chromophores.

Usage
-----
Edit the CONFIG section (ROOT_DIR, FNAME_FLAG, CMEAS_FLAG, cfg_list, etc.)
and run::

        python FIG8_ballsqueezing_images.py

Configurables (defaults shown)
-----------------------------
- ROOT_DIR (str): path to dataset (default used below)
- REC_STR (str): 'conc_o'         # record string (not directly modified here)
- TASK (str): 'BS'                # task identifier used in file IDs
- NOISE_MODEL (str): 'ar_irls'    # used in file naming
- FNAME_FLAG (str): 'mag'         # part of input filename flag - either 'mag' or 'ts'
- C_MEAS_FLAG (bool): True        # part of input filename flag
- T_WIN (list): [5, 8]           # time window (s) to average when using 'ts' inputs
- PLOT_SAVE (bool): False         # whether to save produced figures
- SPATIAL_SMOOTHING (bool): True  # whether to append smoothing suffix to filenames
- SIGMA_SMOOTHING (int): 50       # smoothing kernel size (for filename only)

Visualization options
---------------------
- SCALE (float): 1                  # scaling factor for color limits
- FLAG_IMG_LIST: which image types to render (e.g., 'tstat', 'mag', 'noise')
- FLAG_BRAIN_LIST: whether to plot brain (True) or scalp (False) surfaces
- FLAG_HBO_LIST: whether to display HbO (True) or HbR (False)
- FLAG_CONDITION_LIST: which trial conditions to visualize (e.g., 'right', 'left')
- cfg_list (list[dict]): regularization configurations evaluated (see code
    for defaults: alpha_meas, alpha_spatial, DIRECT, SB, sigma_brain, sigma_scalp)

Outputs
-------
- Optional PNG screenshots saved under
    <ROOT_DIR>/derivatives/cedalion/plots/image_space/<config>/.

Dependencies
------------
- pyvista, matplotlib, xarray, numpy, and project helpers loaded via
    the modules/ directory (spatial_basis_funs, image_recon_func).

Author: Laura Carlton
"""
# %%
import os
import sys
import gzip
import pickle

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

import cedalion.dataclasses as cdc
from cedalion.io.forward_model import load_Adot

pv.set_jupyter_backend("static")

# import my own functions from a different directory
sys.path.append("/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/")
import spatial_basis_func as sbf  # noqa: E402
import image_recon_func as irf  # noqa: E402

# %% set up config parameters
ROOT_DIR = os.path.join("/projectnb", "nphfnirs", "s", "datasets", "BSMW_Laura_Miray_2025", "BS_bids_v2")
REC_STR = "conc_o"
TASK = "BS"
NOISE_MODEL = "ar_irls"
FNAME_FLAG = "mag"
C_MEAS_FLAG = True
PLOT_SAVE = True
T_WIN = [5, 8]
SPATIAL_SMOOTHING = False
SIGMA_SMOOTHING = 50
optional_flag = ''

SCALE = 1
FLAG_HBO_LIST = [True, False] 
FLAG_BRAIN_LIST = [True , False]
FLAG_IMG_LIST = ["tstat", "mag", "stderr", "var_btw", "var_within"]  # , 'noise'] #['mag', 'tstat', 'noise'] #, 'noise'
FLAG_CONDITION_LIST = ["right", "left"] 
lambda_R = 1e-6 

cfg_list = [
  {"alpha_meas": 1e4, "alpha_spatial": 1e-3, "lambda_R": lambda_R, "DIRECT": False, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-3, "lambda_R": lambda_R, "DIRECT": True, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-2, "lambda_R": lambda_R, "DIRECT": False, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-2, "lambda_R": lambda_R, "DIRECT": True, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
   
   {"alpha_meas": 1e2, "alpha_spatial": 1e-3, "lambda_R": lambda_R, "DIRECT": False, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e2, "alpha_spatial": 1e-3, "lambda_R": lambda_R, "DIRECT": True, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e2, "alpha_spatial": 1e-2, "lambda_R": lambda_R, "DIRECT": False, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e2, "alpha_spatial": 1e-2, "lambda_R": lambda_R, "DIRECT": True, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
   
   {"alpha_meas": 1e0, "alpha_spatial": 1e-3, "lambda_R": lambda_R, "DIRECT": False, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e0, "alpha_spatial": 1e-3, "lambda_R": lambda_R, "DIRECT": True, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e0, "alpha_spatial": 1e-2, "lambda_R": lambda_R, "DIRECT": False, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e0, "alpha_spatial": 1e-2, "lambda_R": lambda_R, "DIRECT": True, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
]

if SPATIAL_SMOOTHING:
    smoothing_name = f"_smoothing-{SIGMA_SMOOTHING}"
else:
    smoothing_name = ""

DATA_DIR = os.path.join(ROOT_DIR, "derivatives", "cedalion", "processed_data", "image_space")
SAVE_DIR = os.path.join(ROOT_DIR, "derivatives", "cedalion", "figures", "image_space")

os.makedirs(SAVE_DIR, exist_ok=True)

PROBE_DIR = os.path.join(ROOT_DIR, "derivatives", "cedalion", "fw", "probe")

# %% load head model
head, PARCEL_DIR = irf.load_head_model("ICBM152", with_parcels=True)
Adot = load_Adot(os.path.join(PROBE_DIR, "Adot.nc"))

# %% build plots
threshold = -2  # log10 absolute
wl_idx = 1
M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)

surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
surf = pv.wrap(surf.mesh)

for cfg in cfg_list:
    all_trial_X_hrf_mag = None
    # pdb.set_trace()
    F = None
    D = None
    G = None

    DIRECT = cfg["DIRECT"]
    SB = cfg["SB"]
    sigma_brain = cfg["sigma_brain"]
    sigma_scalp = cfg["sigma_scalp"]
    alpha_meas = cfg["alpha_meas"]
    alpha_spatial = cfg["alpha_spatial"]
    lambda_R = cfg["lambda_R"]

    if DIRECT:
        direct_name = "direct"
    else:
        direct_name = "indirect"

    if C_MEAS_FLAG:
        Cmeas_name = "Cmeas"
    else:
        Cmeas_name = "noCmeas"

    # if SB:
    #     filepath = os.path.join(
    #         DATA_DIR,
    #         f"task-{TASK}_image_hrf_{FNAME_FLAG}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_Cmeas_{NOISE_MODEL}{smoothing_name}{optional_flag}.pkl.gz",
    #     )
    # else:
    #     filepath = os.path.join(
    #         DATA_DIR,
    #         f"task-{TASK}_image_hrf_{FNAME_FLAG}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_Cmeas_{NOISE_MODEL}{smoothing_name}{optional_flag}.pkl.gz",
    #     )
    if SB:
        filepath = os.path.join(
            DATA_DIR,
            f"task-{TASK}_image_hrf_{FNAME_FLAG}_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_Cmeas_{NOISE_MODEL}{smoothing_name}{optional_flag}.pkl.gz",
        )
    else:
        filepath = os.path.join(
            DATA_DIR,
            f"task-{TASK}_image_hrf_{FNAME_FLAG}_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_{direct_name}_Cmeas_{NOISE_MODEL}{smoothing_name}{optional_flag}.pkl.gz",
        )


    with gzip.open(filepath, "rb") as f:
        results = pickle.load(f)

    all_trial_X_tstat = results["X_tstat"]
    all_trial_X_hrf_mag_weighted = results["X_hrf_ts_weighted"]
    all_trial_X_stderr = results["X_std_err"]
    all_trial_X_btw= results["X_mse_between"]
    all_trial_X_within = results["X_mse_within"]
    if FNAME_FLAG == "ts":
        all_trial_X_tstat = all_trial_X_tstat.sel(time=slice(T_WIN[0], T_WIN[1])).mean("time")
        all_trial_X_hrf_mag_weighted = all_trial_X_hrf_mag_weighted.sel(time=slice(T_WIN[0], T_WIN[1])).mean("time")
        all_trial_X_stderr = all_trial_X_stderr.sel(time=slice(T_WIN[0], T_WIN[1])).mean("time")

    for flag_condition in FLAG_CONDITION_LIST:

        for flag_img in FLAG_IMG_LIST:

            for flag_brain in FLAG_BRAIN_LIST:

                for flag_hbo in FLAG_HBO_LIST:

                    p = pv.Plotter(shape=(1, 1), window_size=[1000, 1000], off_screen=PLOT_SAVE)

                    if flag_img == "tstat":
                        foo_img = all_trial_X_tstat.sel(trial_type=flag_condition).copy()
                        title_str = "t-stat"
                    elif flag_img == "mag":
                        foo_img = all_trial_X_hrf_mag_weighted.sel(trial_type=flag_condition).copy()
                        title_str = "magnitude"
                    elif flag_img == "stderr":
                        foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
                        title_str = "std_err"
                    elif flag_img == "var_btw":
                        foo_img = all_trial_X_btw.sel(trial_type=flag_condition).copy()
                        title_str = "between subj var"
                    elif flag_img == "var_within":
                        foo_img = all_trial_X_within.sel(trial_type=flag_condition).copy()
                        title_str = "within subj var"
                    foo_img = foo_img.pint.dequantify()
                    foo_img = foo_img.transpose("chromo", "vertex")

                    if flag_brain:
                        title_str += " brain "
                        surface = "brain"
                        foo_img = foo_img[:, Adot.is_brain.values]
                        foo_img[:, ~M[M.is_brain.values]] = np.nan
                        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
                        surf = pv.wrap(surf.mesh)

                    else:
                        title_str += " scalp "
                        surface = "scalp"
                        foo_img = foo_img[:, ~Adot.is_brain.values]
                        foo_img[:, ~M[~M.is_brain.values]] = np.nan
                        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
                        surf = pv.wrap(surf.mesh)

                    masked = foo_img.sel(chromo="HbO")
                    masked = masked.where(np.isfinite(masked))
                    if flag_img == 'stderr' or flag_img == 'var_btw' or flag_img == 'var_within':
                        clim = (masked.min(skipna=True).values * SCALE, masked.max(skipna=True).values * SCALE)
                    else:
                        clim = (-masked.max(skipna=True).values * SCALE, masked.max(skipna=True).values * SCALE)

                    if flag_hbo:
                        title_str = title_str + flag_condition + ": HbO"
                        foo_img = foo_img.sel(chromo="HbO")
                        chromo = "HbO"
                    else:
                        title_str =  title_str + flag_condition + ": HbR"
                        foo_img = foo_img.sel(chromo="HbR")
                        chromo = "HbR"

                    title_str += f' lambda_R = {lambda_R:0.1e}'
                    p.subplot(0, 0)
                    p.add_mesh(
                        surf,
                        scalars=foo_img,
                        cmap=plt.cm.jet,
                        clim=clim,
                        show_scalar_bar=True,
                        smooth_shading=True,
                        nan_color=(0.9, 0.9, 0.9),
                    )
                    p.camera_position = "xy"
                    p.add_text(title_str, position="lower_left", font_size=10)

                    if PLOT_SAVE:
                        if SB:
                            img_folder = f"images_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{Cmeas_name}"
                        else:
                            img_folder = f"images_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_{direct_name}_{Cmeas_name}"

                        save_dir_tmp = os.path.join(SAVE_DIR, img_folder)
                        if not os.path.exists(save_dir_tmp):
                            os.makedirs(save_dir_tmp)

                        file_name = f"IMG_task-{TASK}_{FNAME_FLAG}_{flag_condition}_{flag_img}_{chromo}_{surface}_scale-{SCALE}{smoothing_name}{optional_flag}.png"
                        p.screenshot(os.path.join(save_dir_tmp, file_name))
                        p.close()
                    else:
                        p.show()


# %%
