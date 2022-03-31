#!/bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=arrykrish@gmail.com
#SBATCH --time=11:30:00
#SBATCH --job-name=sampling-exact
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=medium
#SBATCH --cluster=htc
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/%j.out

module purge
module load CUDA/11.3.1
module load Anaconda3
export CONPREFIX=$DATA/pytorch-env39
source activate $CONPREFIX

echo Sampling the posterior distribution.

python main.py -e False -m exact -f exact_2

echo Sampling completed.
