#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=1:30:00
#SBATCH --job-name=dyadic_exp
#SBATCH --output=slurm_%A-%a.out
#SBATCH --mem=24G
#SBATCH --cpus-per-task=100
#SBATCH --array=1-100%20

# load python module
module load python/3.7.3
# module load ruse

# create a temporary directory for this job and save the name
num=`printf "%03d" ${SLURM_ARRAY_TASK_ID}`
tempdir=/flash/FroeseU/dyadic_exp_${num}

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/Code/dyadic_interaction
source env/bin/activate
python -m dyadic_interaction.main --dir $tempdir --seed $SLURM_ARRAY_TASK_ID --cores 96

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir deigo:/bucket/FroeseU/fede/

# Clean up by removing our temporary directory
rm -r $tempdir*