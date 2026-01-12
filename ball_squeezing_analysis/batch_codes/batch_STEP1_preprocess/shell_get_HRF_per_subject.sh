#!/bin/bash -l
#$ -P nphfnirs
#$ -N  BS_HRF_estimate
#$ -pe omp 16
#$ -l h_rt=12:00:00

#$ -o /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/ball_squeezing_analysis/batch_codes/batch_STEP1_preprocess/output/
#$ -e /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/ball_squeezing_analysis/batch_codes/batch_STEP1_preprocess/error/

# ---------------------------------------------------------------------------
# shell_get_HRF_per_subject.sh
#
# Shell wrapper script to run the per-subject HRF estimation on the cluster
# for the ball-squeezing fNIRS dataset analysis pipeline.
#
# This script is the SGE/qsub wrapper that is submitted by submit_do_hrf_estimation.py.
# It activates the conda environment, sets up PYTHONPATH, and calls the
# per-subject Python worker script `estimate_HRF_per_subj.py` with a single
# subject ID (BIDS-style, e.g. `sub-618`).
#
# Usage
# -----
# This script is typically called automatically by submit_do_hrf_estimation.py,
# but can be submitted manually:
#   qsub shell_get_HRF_per_subject.sh sub-618
#
# Expected environment
# --------------------
# - Assumes `module` system is available and a conda environment named
#   `cedalion_snakemake` is present.
# - Logs are written to the output/error directories specified by the SGE directives.
# - The script expects one positional argument: the subject ID.
#
# Outputs
# -------
# - Stdout/stderr are redirected to the SGE -o/-e paths defined above.
# - The called Python script will write per-subject HRF estimates and preprocessed
#   data as gzipped pickles under <ROOT_DIR>/derivatives/cedalion/processed_data/<sub>/.
#
# Author: Laura Carlton
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

# Minimal argument check so the job fails fast with a helpful message
if [ -z "${subject}" ]; then
	echo "Usage: $0 <subject_id>"
	echo "Example: qsub $0 sub-618"
	exit 1
fi

echo "subject : ${subject}"
echo "=========================================================="

python /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/ball_squeezing_analysis/batch_codes/batch_STEP1_preprocess/estimate_HRF_per_subj.py $subject
