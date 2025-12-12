#!/bin/bash -l
#$ -P nphfnirs
#$ -N  augment_data
#$ -pe omp 16
#$ -l h_rt=12:00:00

#$ -o /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/augmentation/output/
#$ -e /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/augmentation/error/


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

python /projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/augmented_simulation/batch_codes/augmentation/get_measurement_variance_per_subject.py $subject



