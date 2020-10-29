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

# create a temporary directory for this job and save the name
tempdir=/flash/FroeseU/georgina/dyadic_exp_1

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/Code/dyadic_interaction
source env/bin/activate
# ruse
# python -m dyadic_interaction.main --entropy shannon --dir $tempdir --seed 1 --cores 96
python -m dyadic_interaction.main --entropy transfer --dir $tempdir --seed 1 --cores 96 --perf_obj MAX

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir deigo:/bucket/FroeseU/georgina/gitlab/tmp_exp

# Clean up by removing our temporary directory
rm -r $tempdir