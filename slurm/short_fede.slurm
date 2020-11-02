#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --job-name=dyadic_exp
#SBATCH --mem=500G
#SBATCH --cpus-per-task=128
##SBATCH --mail-type=ALL
##SBATCH --mail-user=federico.sangati2@oist.jp

# load python module
# module load ruse
module load python/3.7.3
module load java-jdk/14
# module load ruse

# select the seed
seed=1

# create a temporary directory for this job and save the name
dir_seed=`printf "%03d" ${seed}`
tempdir=/flash/FroeseU/fede/seed_${dir_seed}

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/Code/dyadic_interaction
source env/bin/activate

# ruse
python -m dyadic_interaction.main \
--dir $tempdir \
--entropy_type shannon \
--entropy_target_value neural_outputs \
--concatenate on \
--collision_type overlapping \
--cores 96 \
--seed ${seed} \
--num_neurons 2 \
--popsize 96 \
--num_gen 500 \
--trial_duration 200

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir deigo:/bucket/FroeseU/fede/tmp_exp/

# Clean up by removing our temporary directory
rm -r $tempdir