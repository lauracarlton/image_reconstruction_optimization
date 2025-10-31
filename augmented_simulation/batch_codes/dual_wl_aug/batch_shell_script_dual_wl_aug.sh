#!/bin/bash -l
#$ -P nphfnirs
#$ -N  dual_WL_aug
#$ -pe omp 1
#$ -l h_rt=12:00:00

#$ -o /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/dual_wl_aug/output/
#$ -e /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/dual_wl_aug/error/


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

python /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/dual_wl_aug/dual_wl_metrics_batch_aug.py $alpha_meas $alpha_spatial $sigma_brain $sigma_scalp


