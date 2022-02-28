#!/bin/tcsh
#SBATCH --job-name=generate-baseline # Job name
#SBATCH --array=1-50 # NOTE: not offset by one like other scripts, inclusive range
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 20 # 1 cpu on single node
#SBATCH --mem=25gb # Job memory request
#SBATCH --time=120:00:00 # Time limit hrs:min:sec
#SBATCH --mail-type=BEGIN,END,FAIL. # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adhinart@bc.edu # Where to send mail
#SBATCH --partition=partial_nodes,full_nodes48,full_nodes64,gpuv100,gpua100,anzellos

module purge
module load anaconda
conda activate dendrite

cd /mmfs1/data/adhinart/dendrite
setenv TMPDIR /scratch/adhinart/dendrite/$SLURM_ARRAY_TASK_ID
rm -rf $TMPDIR
mkdir -p $TMPDIR
cp extracted/$SLURM_ARRAY_TASK_ID.h5 $TMPDIR

python3 chunk_sphere.py $TMPDIR $SLURM_ARRAY_TASK_ID
cp $TMPDIR/seg_$SLURM_ARRAY_TASK_ID.h5 baseline/

rm -rf $TMPDIR
