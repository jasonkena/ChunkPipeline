#!/bin/tcsh
#SBATCH --job-name=dendrite_stats # Job name
#SBATCH --nodes=1
#SBATCH --ntasks 1
#SBATCh --cpus-per-task 48 # 1 cpu on single node
#SBATCH --mem=30gb # Job memory request
#SBATCH --time=02:00:00 # Time limit hrs:min:sec
#SBATCH --mail-type=BEGIN,END,FAIL. # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adhinart@bc.edu # Where to send mail
#SBATCH --partition=partial_nodes,full_nodes48,full_nodes64,gpuv100,gpua100

setenv BASE_PATH mouse

cd /mmfs1/data/adhinart/dendrite
module load anaconda
conda activate dendrite

python evaluation.py $BASE_PATH
