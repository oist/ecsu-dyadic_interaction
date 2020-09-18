#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=5:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=100
##SBATCH --mail-type=ALL
##SBATCH --mail-user=federico.sangati2@oist.jp

# load python module
module load python/3.7.3
# module load ruse

# create a temporary directory for this job and save the name
tempdir=/flash/FroeseU/dydadic_exp

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/Code/dyadic_interaction
source env/bin/activate
python -m dyadic_interaction.main --dir $tempdir --cores 96

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir* deigo:/bucket/FroeseU/fede/

# Clean up by removing our temporary directory
rm -r $tempdir*