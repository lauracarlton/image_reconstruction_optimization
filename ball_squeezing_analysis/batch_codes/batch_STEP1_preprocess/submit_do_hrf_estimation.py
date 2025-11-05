#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submit_do_hrf_estimation.py

Convenience script to submit per-subject HRF estimation jobs to the cluster
queue used in the imaging paper analysis pipeline.

What it does
------------
- Scans `root_dir` for subject folders (names containing 'sub').
- Skips any subjects listed in `excluded`.
- For each remaining subject it builds a qsub command that calls the
    project shell script `shell_get_HRF_per_subject.sh` with the subject id and
    submits it using `subprocess.run`.

Usage
-----
Edit the configuration variables below (ROOT_DIR, EXCLUDED, SHELL_SCRIPT)
then run from a login node where qsub is available:

        python submit_do_hrf_estimation.py

Configurables
-------------
- ROOT_DIR (str): path to the BIDS-like dataset containing subject folders.
- EXCLUDED (list[str]): subject IDs to skip (default contains a few known
    problematic subjects).
- SHELL_SCRIPT (str): absolute path to the shell wrapper that launches the
    per-subject HRF estimation job (the script passed to qsub).

Notes
-----
- This script assumes a Sun Grid Engine-style `qsub` command is available on
    the PATH. If your cluster uses a different scheduler, update `qsub_cmd`
    accordingly.
- The script uses `subprocess.run(..., shell=True)` to run the qsub command

Author: Laura Carlton
"""

import os
import subprocess

# --------------------- CONFIG ---------------------
ROOT_DIR = os.path.join("/projectnb", "nphfnirs", "s", "datasets", "BSMW_Laura_Miray_2025", "BS_bids")
EXCLUDED = []
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
    "batch_STEP1_preprocess",
    "shell_get_HRF_per_subject.sh",
)
# --------------------------------------------------

dirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
subject_list = [d for d in dirs if "sub" in d and d not in EXCLUDED]

for subj in subject_list:
    qsub_command = f"qsub {SHELL_SCRIPT} {subj}"
    subprocess.run(qsub_command, shell=True)
