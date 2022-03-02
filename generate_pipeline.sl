#!/bin/tcsh -e
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

setenv BASE_PATH mouse
setenv TMPDIR /scratch/adhinart/dendrite/$SLURM_ARRAY_TASK_ID/$BASE_PATH

rm -rf $TMPDIR
mkdir -p $TMPDIR

cd /mmfs1/data/adhinart/dendrite/$BASE_PATH
mkdir -p extracted
# for points
mkdir -p results 
mkdir -p baseline

cp *.h5 $TMPDIR
cp *.npy $TMPDIR

cd ..

if ( -f "$BASE_PATH/extracted/$SLURM_ARRAY_TASK_ID.h5" ) then
    echo Extraction already exists
    cp $BASE_PATH/extracted/$SLURM_ARRAY_TASK_ID.h5 $TMPDIR
else
    python3 extract_seg.py $TMPDIR $SLURM_ARRAY_TASK_ID
    cp $TMPDIR/$SLURM_ARRAY_TASK_ID.h5 $BASE_PATH/extracted/
endif

echo extract_seg finished
if ( -f "$BASE_PATH/results/$SLURM_ARRAY_TASK_ID.npy" ) then
    echo Points already exist
else
    python3 point.py $TMPDIR $SLURM_ARRAY_TASK_ID
    cp $TMPDIR/$SLURM_ARRAY_TASK_ID.npy $BASE_PATH/results/
endif
echo point_generation finished

if ( -f "$BASE_PATH/baseline/seg_$SLURM_ARRAY_TASK_ID.h5" ) then
    echo Baseline already exists
else
    python3 chunk_sphere.py $TMPDIR $SLURM_ARRAY_TASK_ID
    cp $TMPDIR/seg_$SLURM_ARRAY_TASK_ID.h5 $BASE_PATH/baseline/
endif
echo baseline finished

rm -rf $TMPDIR
