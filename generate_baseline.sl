#!/bin/tcsh
#SBATCH --job-name=generate-points # Job name
#SBATCH --array=26-28 # NOTE: not offset by one like other scripts, inclusive range
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 20 # 1 cpu on single node
#SBATCH --mem=30gb # Job memory request
#SBATCH --time=120:00:00 # Time limit hrs:min:sec
#SBATCH --mail-type=BEGIN,END,FAIL. # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adhinart@bc.edu # Where to send mail
#SBATCH --partition=partial_nodes,full_nodes48,full_nodes64,gpuv100,gpua100,anzellos

module purge
module load anaconda
conda activate dendrite

cd /mmfs1/data/adhinart/dendrite

python3 point.py ./extracted $SLURM_ARRAY_TASK_ID
