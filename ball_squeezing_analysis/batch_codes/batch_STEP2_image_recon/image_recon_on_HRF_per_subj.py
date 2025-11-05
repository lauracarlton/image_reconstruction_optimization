#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_recon_on_HRF_per_subj.py

Run image reconstruction for a single subject using previously-estimated
HRFs. This script is the per-subject worker called by the FIG8 image-recon
batch submitter. It reads the saved per-subject HRF and MSE pickles produced
by the HRF estimation step, runs a small set of reconstruction configurations
(`cfg_list`) and writes per-subject image-space HRF and MSE results as
gzipped pickles under the dataset's derivatives directory.

Usage
-----
- Configure the runtime variables in the CONFIG section below (ROOT_DIR,
    PROBE_DIR, NOISE_MODEL, flags, regularization configs).
- Run from the command line with a single subject id (BIDS-style):

        python image_recon_on_HRF_per_subj.py sub-618

Inputs
------
- Root dataset folder containing per-subject `nirs` subfolders and STEP1
    outputs located under `<ROOT_DIR>/derivatives/processed_data/<subject>/`.
- Forward model `Adot.nc` in the PROBE_DIR used for sensitivity and masking.

Configurables (top of file)
---------------------------
- ROOT_DIR (str): '/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids'
        - Path to BIDS-like dataset used by STEP1 outputs and subject SNIRF files.
- NOISE_MODEL (str): 'ar_irls'
        - Noise-model label used for solving the GLM when reading per-subject STEP1 outputs.
- REC_STR (str): 'conc_o'
        - Record string pointing to concentration data used by STEP1.
- - TASK (str): 'BS'
    - Task identifier used to build file IDs.
- CMEAS_FLAG (bool): True
    - Whether to use measured C_meas when performing image reconstruction.
- MAG_TS_FLAG (str): 'MAG'  # expected values: 'MAG' or 'TS'
        - Controls whether to reduce HRF to a single magnitude value (MAG) or
            keep time series (TS) when reconstructing.
- T_WIN (list[int]): [5, 8]
        - Time window (seconds) used for computing magnitude when MAG flag is set.
- cfg_mse (dict): keys for MSE masking and thresholds (see script for defaults).
- cfg_list (list[dict]): Regularization configurations evaluated (alpha_meas,
    alpha_spatial, DIRECT, SB, sigma_brain, sigma_scalp).

Outputs
-------
- For each subject and configuration, a gzipped pickle saved to:
    `<ROOT_DIR>/derivatives/processed_data/image_space/<subject>/` containing:
    - 'X_hrf' : reconstructed image(s) (xarray)
    - 'X_mse' : per-image MSE estimates (xarray)

Dependencies
------------
Requires the project `cedalion` package and the local helper modules in
the project's modules directory (image_recon_func, spatial_basis_funs).
Also requires xarray, numpy and python's gzip/pickle.


Author: Laura Carlton
"""

# %%
import os
import sys
import gzip
import pickle
import warnings

import xarray as xr
from cedalion import nirs, units, io
from cedalion.io.forward_model import load_Adot

# import functions from a different directory
sys.path.append("/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/")
import image_recon_func as irf  # noqa: E402

# Turn off all warnings
warnings.filterwarnings("ignore")

# %%
subject = str(sys.argv[1])
# subject = 'sub-580'
# %% set up config parameters
ROOT_DIR = os.path.join("/", "projectnb", "nphfnirs", "s", "datasets", "BSMW_Laura_Miray_2025", "BS_bids")
NOISE_MODEL = "ar_irls"
TASK = "BS"
REC_STR = "conc_o"
CMEAS_FLAG = True
MAG_TS_FLAG = "mag"
T_WIN = [5, 8]

cfg_list = [
    {"alpha_meas": 1e4, "alpha_spatial": 1e-3, "DIRECT": False, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e2, "alpha_spatial": 1e-3, "DIRECT": True, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-2, "DIRECT": False, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e2, "alpha_spatial": 1e-2, "DIRECT": True, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
]

cfg_mse = {"mse_val_for_bad_data": 1e1, "mse_amp_thresh": 1e-3 * units.V, "blockaverage_val": 0, "mse_min_thresh": 1e-6}

SAVE_DIR = os.path.join(ROOT_DIR, "derivatives", "cedalion", "processed_data", "image_space", subject)
PROBE_DIR = os.path.join(ROOT_DIR, "derivatives", "cedalion", "fw", "ICBM152")
os.makedirs(SAVE_DIR, exist_ok=True)

# %% load head model
head, PARCEL_DIR = irf.load_head_model("ICBM152", with_parcels=True)
Adot = load_Adot(os.path.join(PROBE_DIR, "Adot.nc"))

recordings = io.read_snirf(os.path.join(ROOT_DIR, f"{subject}/nirs/{subject}_task-{TASK}_run-01_nirs.snirf"))
rec = recordings[0]
geo3d = rec.geo3d

print("Loading saved data")
with gzip.open(
    os.path.join(
        ROOT_DIR,
        "derivatives",
        "cedalion",
        "processed_data",
        subject,
        f"{subject}_task-{TASK}_{REC_STR}_hrf_estimates_{NOISE_MODEL}.pkl.gz",
    ),
    "rb",
) as f:
    all_results = pickle.load(f)

# %% run image recon
"""
do the image reconstruction of each subject independently
- this is the unweighted subject block average magnitude
- then reconstruct their individual MSE
- then get the weighted average in image space
- get the total standard error using between + within subject MSE
"""

subj_hrf = all_results["hrf_per_subj"]
subj_mse = all_results["hrf_mse_per_subj"]
bad_channels = all_results["bad_indices"]

dpf = xr.DataArray(
    [1, 1],
    dims="wavelength",
    coords={"wavelength": Adot.wavelength},
)
E = nirs.get_extinction_coefficients("prahl", Adot.wavelength)

# %%
for cfg in cfg_list:
    F = None
    D = None

    DIRECT = cfg["DIRECT"]
    SB = cfg["SB"]

    sigma_brain = cfg["sigma_brain"]
    sigma_scalp = cfg["sigma_scalp"]
    alpha_meas = cfg["alpha_meas"]
    alpha_spatial = cfg["alpha_spatial"]

    if os.path.exists(PROBE_DIR + f"G_matrix_sigmabrain-{float(sigma_brain)}.pkl") and os.path.exists(
        PROBE_DIR + f"G_matrix_sigmascalp-{float(sigma_scalp)}.pkl"
    ):
        G_EXISTS = True
        with open(PROBE_DIR + f"G_matrix_sigmabrain-{float(sigma_brain)}.pkl", "rb") as f:
            G_brain = pickle.load(f)

        with open(PROBE_DIR + f"G_matrix_sigmascalp-{float(sigma_scalp)}.pkl", "rb") as f:
            G_scalp = pickle.load(f)

        G = {"G_brain": G_brain, "G_scalp": G_scalp}

    else:
        G_EXISTS = False
        G = None

    if DIRECT:
        direct_name = "direct"
    else:
        direct_name = "indirect"

    if CMEAS_FLAG:
        Cmeas_name = "Cmeas"
    else:
        Cmeas_name = "noCmeas"

    cfg_sbf = {
        "mask_threshold": -2,
        "threshold_brain": sigma_brain * units.mm,
        "threshold_scalp": sigma_scalp * units.mm,
        "sigma_brain": sigma_brain * units.mm,
        "sigma_scalp": sigma_scalp * units.mm,
    }

    print(f"alpha_meas = {alpha_meas}, alpha_spatial = {alpha_spatial}, SB = {SB}, {direct_name}")

    all_trial_X_hrf = None
    all_trial_X_mse = None

    print(f"\tCalculating subject = {subject}")

    for trial_type in subj_hrf.trial_type:

        print(f"\t\tGetting images for trial type = {trial_type.values}")

        hrf = subj_hrf.squeeze().sel(trial_type=trial_type)

        od_hrf = nirs.conc2od(hrf, geo3d, dpf)

        mse = subj_mse.squeeze().sel(trial_type=trial_type)
        od_mse = xr.dot(E**2, mse, dim=["chromo"]) * 1 * units.mm**2

        channels = od_hrf.channel
        od_mse.loc[:, channels.isel(channel=bad_channels), :] = cfg_mse["mse_val_for_bad_data"]
        od_mse = xr.where(
            od_mse < cfg_mse["mse_min_thresh"], cfg_mse["mse_min_thresh"], od_mse
        )  # !!! maybe can be removed when we have the between subject mse
        od_hrf.loc[:, channels.isel(channel=bad_channels), :] = cfg_mse["blockaverage_val"]

        od_mse_mag = od_mse.mean("time")

        if MAG_TS_FLAG == "MAG":
            od_hrf = od_hrf.sel(time=slice(T_WIN[0], T_WIN[1])).mean("time")
            fname_flag = "mag"
        else:
            fname_flag = "ts"

        C_meas = od_mse_mag.pint.dequantify()
        C_meas = C_meas.stack(measurement=("channel", "wavelength")).sortby("wavelength")

        X_hrf, W, D, F, G = irf.do_image_recon(
            od_hrf,
            head=head,
            Adot=Adot,
            C_meas_flag=CMEAS_FLAG,
            C_meas=C_meas,
            wavelength=[760, 850],
            BRAIN_ONLY=False,
            DIRECT=DIRECT,
            SB=SB,
            cfg_sbf=cfg_sbf,
            alpha_spatial=alpha_spatial,
            alpha_meas=alpha_meas,
            F=F,
            D=D,
            G=G,
        )

        if SB and not G_EXISTS:
            with open(PROBE_DIR + f"G_matrix_sigmabrain-{float(sigma_brain)}.pkl", "wb") as f:
                pickle.dump(G["G_brain"], f)

            with open(PROBE_DIR + f"G_matrix_sigmascalp-{float(sigma_scalp)}.pkl", "wb") as f:
                pickle.dump(G["G_scalp"], f)

            G_EXISTS = True

        od_mse = od_mse.stack(measurement=("channel", "wavelength")).sortby("wavelength")
        od_mse = od_mse.transpose("measurement", "time")
        if MAG_TS_FLAG == "MAG":
            template = X_hrf
        else:
            template = X_hrf.isel(time=0).squeeze()

        X_mse = irf.get_image_noise(C_meas, template, W, DIRECT=DIRECT, SB=SB, G=G)

        if all_trial_X_hrf is None:

            all_trial_X_hrf = X_hrf
            all_trial_X_hrf = all_trial_X_hrf.assign_coords(subject=subject)
            all_trial_X_hrf = all_trial_X_hrf.assign_coords(trial_type=trial_type)

            all_trial_X_mse = X_mse
            all_trial_X_mse = all_trial_X_mse.assign_coords(subject=subject)
            all_trial_X_mse = all_trial_X_mse.assign_coords(trial_type=trial_type)

        else:

            X_hrf = X_hrf.assign_coords(subject=subject)
            X_hrf = X_hrf.assign_coords(trial_type=trial_type)

            X_mse_tmp = X_mse.assign_coords(subject=subject)
            X_mse_tmp = X_mse_tmp.assign_coords(trial_type=trial_type)

            all_trial_X_hrf = xr.concat([all_trial_X_hrf, X_hrf], dim="trial_type")
            all_trial_X_mse = xr.concat([all_trial_X_mse, X_mse_tmp], dim="trial_type")

    results = {"X_hrf": all_trial_X_hrf, "X_mse": all_trial_X_mse}

    print(f"\t\tSaving to {SAVE_DIR}")

    if SB:
        filepath = os.path.join(
            SAVE_DIR,
            f"{subject}_task-{TASK}_image_hrf_{fname_flag}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{Cmeas_name}_{NOISE_MODEL}.pkl.gz",
        )
    else:
        filepath = os.path.join(
            SAVE_DIR,
            f"{subject}_task-{TASK}_image_hrf_{fname_flag}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_{Cmeas_name}_{NOISE_MODEL}.pkl.gz",
        )

    file = gzip.GzipFile(filepath, "wb")
    file.write(pickle.dumps(results))
    file.close()

print("Job Complete")
# %%
