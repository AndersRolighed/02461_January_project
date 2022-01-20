#!/bin/sh
#BSUB -J 300-bce
#BSUB -o 300-bce_%J.out
#BSUB -e 300-bce_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
# end of BSUB options

module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
module load ffmpeg/4.2.2
module load openblas/0.2.20
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20

# activate the virtual environment
source jan-env/bin/activate

python3 training_hpc_ba2-lr0001-ep300-bce.py
