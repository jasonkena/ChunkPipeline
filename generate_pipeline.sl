#!/bin/tcsh
#SBATCH --job-name=generate-points # Job name
#SBATCH --array=1-50 # inclusive range
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
module purge
module load anaconda
conda activate dendrite

cd /mmfs1/data/adhinart/dendrite
mkdir -p extracted
# for points
mkdir -p results 
mkdir -p baseline

setenv TMPDIR /scratch/adhinart/dendrite/$SLURM_ARRAY_TASK_ID
rm -rf $TMPDIR
mkdir -p $TMPDIR
cp *.txt $TMPDIR
cp *.h5 $TMPDIR
cp *.npy $TMPDIR

python3 extract_seg.py $TMPDIR $SLURM_ARRAY_TASK_ID
echo extract_seg finished
cp $TMPDIR/$SLURM_ARRAY_TASK_ID.h5 extracted/
cp extracted/$SLURM_ARRAY_TASK_ID.h5 $TMPDIR
python3 point.py $TMPDIR $SLURM_ARRAY_TASK_ID
echo point_generation finished
python3 chunk_sphere.py $TMPDIR $SLURM_ARRAY_TASK_ID
echo baseline finished
cp $TMPDIR/seg_$SLURM_ARRAY_TASK_ID.h5 baseline/

rm -rf $TMPDIR
