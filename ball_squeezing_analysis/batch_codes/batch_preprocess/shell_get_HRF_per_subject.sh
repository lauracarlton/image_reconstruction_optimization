#!/bin/bash -l
#$ -P nphfnirs
#$ -N  BS_HRF_estimate
#$ -pe omp 16
#$ -l h_rt=12:00:00

#$ -o /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/ball_squeezing_analysis/batch_codes/batch_preprocess/output/
#$ -e /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/ball_squeezing_analysis/batch_codes/batch_preprocess/error/


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

python /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/ball_squeezing_analysis/batch_codes/batch_preprocess/estimate_HRF_per_subj.py $subject
