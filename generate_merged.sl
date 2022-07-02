#!/bin/tcsh
#SBATCH --job-name=generate-points # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 20 # 1 cpu on single node
#SBATCH --mem=30gb # Job memory request
#SBATCH --time=120:00:00 # Time limit hrs:min:sec
#SBATCH --mail-type=BEGIN,END,FAIL. # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adhinart@bc.edu # Where to send mail
#SBATCH --partition=partial_nodes,full_nodes48,full_nodes64,gpuv100,gpua100

# assumes that den_6nm_bb.npy exists
# entire pipeline for generating files base h5 up to the final segmentation, + point generation
# merge has to be run manually
setenv BASE_PATH mouse

module purge
module load anaconda
conda activate dendrite

cd /mmfs1/data/adhinart/dendrite

python3 merge.py $BASE_PATH
