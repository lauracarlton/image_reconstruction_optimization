#!/bin/bash -l
#$ -P nphfnirs
#$ -N  BS_IR
#$ -pe omp 16
#$ -l h_rt=12:00:00

#$ -o /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/ball_squeezing_analysis/batch_codes/batch_STEP2_image_recon/output/
#$ -e /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/ball_squeezing_analysis/batch_codes/batch_STEP2_image_recon/error/

# ---------------------------------------------------------------------------
# shell_do_image_recon_on_HRF.sh
#
# Small wrapper script to run the per-subject HRF estimation on the cluster.
# This script is intended to be submitted via the SGE/qsub scheduler. It
# activates the conda environment, sets up PYTHONPATH, and calls the
# per-subject Python runner `image_recon_on_HRF_per_subj.py` with a single
# subject id (BIDS-style, e.g. `sub-618`).
#
# Usage (example):
#   qsub shell_do_image_recon_on_HRF.sh sub-618
#
# Expected environment & notes:
# - Assumes `module` system is available and a conda env named
#   `cedalion_snakemake` is present. Adjust activation to match your site.
# - Logs are written to the paths declared below by the SGE directives.
# - The script expects one positional argument: the subject id.
#
# Outputs:
# - Stdout/stderr are redirected to the SGE -o/-e paths defined above.
# - The called Python script will write per-subject pickles under the
#   project's derivatives/processed_data folder.
# ---------------------------------------------------------------------------

# load miniconda
module load miniconda
conda activate cedalion_snakemake

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID"
echo "=========================================================="


export PYTHONPATH=/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/:$PYTHONPATH
export QT_QPA_PLATFORM=offscreen

subject=${1}

echo "subject : $subject"
echo "=========================================================="

python /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/ball_squeezing_analysis/batch_codes/batch_STEP2_image_recon/image_recon_on_HRF_per_subj.py $subject
