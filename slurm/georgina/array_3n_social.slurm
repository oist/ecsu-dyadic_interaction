#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --job-name=dyadic_exp
#SBATCH --output=slurm_%A-%a.out
#SBATCH --mem=500G
#SBATCH --cpus-per-task=128
#SBATCH --array=1-10%10

# load python module
module load python/3.7.3
module load java-jdk/14
# module load ruse

# create a temporary directory for this job and save the name
seed_dir=${SLURM_JOB_ID}_`printf "%03d" ${SLURM_ARRAY_TASK_ID}`
tempdir=/flash/FroeseU/georgina/${seed_dir}
mkdir ${tempdir}

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/gitlab/dyadic_interaction
source env/bin/activate

# ruse
python -m dyadic_interaction.main \
--dir $tempdir \
--entropy_type shannon \
--entropy_target_value neural \
--concatenate off \
--isolation off \
--collision_type edge \
--cores 96 \
--seed ${SLURM_ARRAY_TASK_ID} \
--num_neurons 3 \
--popsize 96 \
--num_gen 2000 \
--trial_duration 200

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir/* deigo:/bucket/FroeseU/georgina/gitlab

# Clean up by removing our temporary directory
rm -r $tempdir