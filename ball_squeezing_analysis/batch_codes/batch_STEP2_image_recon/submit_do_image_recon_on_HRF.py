#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submit_do_image_recon_on_HRF.py

Convenience script to submit per-subject image reconstruction of the HRF jobs to the cluster
queue used in the imaging paper analysis pipeline.

What it does
------------
- Scans `root_dir` for subject folders (names containing 'sub').
- Skips any subjects listed in `excluded`.
- For each remaining subject it builds a qsub command that calls the
    project shell script `shell_do_image_recon_on_HRF.sh` with the subject id and
    submits it using `subprocess.run`.

Usage
-----
Adjust the CONFIG section below (ROOT_DIR, EXCLUDED, SHELL_SCRIPT) if
needed, then run from the cluster head node or any machine with access to
the qsub command:

    python submit_do_image_recon_on_HRF.py

Configurables
-------------
- ROOT_DIR: path to the BIDS-like dataset root containing subject folders.
- EXCLUDED: list of subject IDs to skip.
- SHELL_SCRIPT: path to the shell wrapper that performs the image recon per subject.

Notes
-----
- This script assumes a Sun Grid Engine-style `qsub` command is available on
    the PATH. If your cluster uses a different scheduler, update `qsub_cmd`
    accordingly.
- The script uses `subprocess.run(..., shell=True)` to run the qsub command

Author: Laura Carlton
"""

# %%
import os
import subprocess

# --------------------- CONFIG ---------------------
ROOT_DIR = os.path.join("/projectnb", "nphfnirs", "s", "datasets", "BSMW_Laura_Miray_2025", "BS_bids")
EXCLUDED = ["sub-538", "sub-549", "sub-547"]

SHELL_SCRIPT = os.path.join(
    "/projectnb",
    "nphfnirs",
    "s",
    "users",
    "lcarlton",
    "ANALYSIS_CODE",
    "imaging_paper_figure_code",
    "ball_squeezing_analysis",
    "batch_codes",
    "batch_STEP2_image_recon",
    "shell_do_image_recon_on_HRF.sh",
)
# --------------------------------------------------

# list entries and filter to directories that look like subjects
dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if "sub" in d and d not in EXCLUDED]

for subj in subject_list:
    qsub_command = f"qsub {SHELL_SCRIPT} {subj}"
    subprocess.run(qsub_command, shell=True)
