#!/bin/bash -l
#$ -P nphfnirs
#$ -N  single_WL_aug
#$ -pe omp 1
#$ -l h_rt=12:00:00

#$ -o /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/single_wl_aug/output/
#$ -e /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/single_wl_aug/error/

# ---------------------------------------------------------------------------
# batch_shell_script_single_wl_aug.sh
#
# Wrapper script to run single-wavelength augmented simulation metric computation
# on the cluster. This script is intended to be submitted via the SGE/qsub
# scheduler. It activates the conda environment, sets up PYTHONPATH, and calls
# the batch worker script `single_wl_metrics_batch_aug.py` with regularization
# and spatial basis function parameters.
#
# Usage (example):
#   qsub batch_shell_script_single_wl_aug.sh 1.0 0.01 0 0
#   qsub batch_shell_script_single_wl_aug.sh 10.0 0.001 1 5
#
# Arguments:
#   $1: alpha_meas - Measurement regularization parameter (e.g., 0.1, 1, 10, 100)
#   $2: alpha_spatial - Spatial regularization parameter (e.g., 1e-3, 1e-2)
#   $3: sigma_brain - Brain spatial basis function width in mm (e.g., 0, 1, 3, 5)
#   $4: sigma_scalp - Scalp spatial basis function width in mm (e.g., 0, 1, 5, 10, 20)
#
# Expected environment & notes:
# - Assumes `module` system is available and a conda env named
#   `cedalion_snakemake` is present. Adjust activation to match your site.
# - Logs are written to the paths declared above by the SGE directives.
# - The script expects four positional arguments as listed above.
#
# Outputs:
# - Stdout/stderr are redirected to the SGE -o/-e paths defined above.
# - The called Python script will write individual metric result pickles under
#   the project's derivatives/cedalion/augmented_data/batch_results/single_wl/ folder.
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

alpha_meas=${1}
alpha_spatial=${2}
sigma_brain=${3}
sigma_scalp=${4}

echo "alpha_meas : $alpha_meas"
echo "alpha_spatial : $alpha_spatial"
echo "sigma_brain : $sigma_brain"
echo "sigma_scalp : $sigma_scalp"
echo "=========================================================="

python /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/single_wl_aug/single_wl_metrics_batch_aug.py $alpha_meas $alpha_spatial $sigma_brain $sigma_scalp


