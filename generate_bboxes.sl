#!/bin/tcsh
#SBATCH --job-name=bbox. # Job name
#SBATCH --ntasks 1 --cpus-per-task 1 # 1 cpu on single node
#SBATCH --mem=50gb # Job memory request
#SBATCH --time=02:00:00 # Time limit hrs:min:sec
#SBATCH --mail-type=BEGIN,END,FAIL. # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adhinart@bc.edu # Where to send mail

cd /mmfs1/data/adhinart/dendrite
module load anaconda
conda activate dendrite

python generate_bboxes.py
