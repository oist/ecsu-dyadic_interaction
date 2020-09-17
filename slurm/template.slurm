#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=0-1
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
##SBATCH --output=slurm-%j.out
##SBATCH --error=slurm-%j.err	
##SBATCH --mail-type=ALL
##SBATCH --mail-user=federico.sangati2@oist.jp

# create a temporary directory for this job and save the name
tempdir=$(mktemp -d /flash/MyunitU/myprog.XXXXXX)

# enter the temporary directory
cd $tempdir

# Start 'myprog' with input from bucket,
# and output to our temporary directory
myprog /bucket/MyunitU/mydata.dat -o output.dat

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
scp output.dat deigo:/bucket/MyunitU/

# Clean up by removing our temporary directory
rm -r $tempdir