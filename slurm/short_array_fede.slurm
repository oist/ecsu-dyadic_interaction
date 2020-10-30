#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=2:00:00
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
seed=`printf "%03d" ${SLURM_ARRAY_TASK_ID}`
tempdir=/flash/FroeseU/fede/seed_${seed}

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/Code/dyadic_interaction
source env/bin/activate
python -m dyadic_interaction.main --entropy shannon --dir $tempdir --seed $SLURM_ARRAY_TASK_ID --cores 96 --num_gen 2000
# python -m dyadic_interaction.main --entropy transfer --dir $tempdir --seed $SLURM_ARRAY_TASK_ID --cores 96 --perf_obj MAX

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir deigo:/bucket/FroeseU/fede/tmp_exp/

# Clean up by removing our temporary directory
rm -r $tempdir