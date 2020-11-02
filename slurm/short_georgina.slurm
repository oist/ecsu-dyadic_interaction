#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=2:00:00
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

# create a temporary directory for this job and save the name
tempdir=/flash/FroeseU/georgina/dyadic_exp_1

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/gitlab/dyadic_interaction
source env/bin/activate
# ruse
#python -m dyadic_interaction.main --dir $tempdir --entropy_type sample --entropy_target_value agents_distance --cores 96 --seed 1 --num_neurons=3 --popsize 96 --num_gen 2000 --trial_duration 200
python -m dyadic_interaction.main --dir $tempdir --entropy_type shannon --entropy_target_value neural --cores 96 --seed 1 --num_neurons=3 --popsize 96 --num_gen 2000 --trial_duration 200


# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir deigo:/bucket/FroeseU/georgina/gitlab/tmp_exp

# Clean up by removing our temporary directory
rm -r $tempdir