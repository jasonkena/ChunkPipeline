#!/bin/tcsh
#SBATCH --job-name=process_tasks. # Job name
#SBATCH --ntasks 1 --cpus-per-task 4 # 1 cpu on single node
#SBATCH --mem=100gb # Job memory request
#SBATCH --time=02:00:00 # Time limit hrs:min:sec
#SBATCH --mail-type=BEGIN,END,FAIL. # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adhinart@bc.edu # Where to send mail

cd /mmfs1/data/adhinart/dendrite
module load anaconda
conda activate dendrite

rq worker --burst -c settings &
rq worker --burst -c settings &
rq worker --burst -c settings &
rq worker --burst -c settings &
wait
