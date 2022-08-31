#!/bin/tcsh -e
#SBATCH --job-name=generate-pipeline # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 48 # 1 cpu on single node
#SBATCH --mem=190gb # Job memory request
#SBATCH --time=120:00:00 # Time limit hrs:min:sec
#SBATCH --mail-type=BEGIN,END,FAIL. # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adhinart@bc.edu # Where to send mail
#SBATCH --partition=partial_nodes,gpuv100,gpua100

#SBATCH --output=main_%j.out 

module purge
module load anaconda
module load slurm
conda activate clean_dendrite

which python
hostname


cd /mmfs1/data/adhinart/dendrite
/data/adhinart/.conda/envs/clean_dendrite/bin/python -m chunk_pipeline.main --pipeline DendritePipeline --config human.py
