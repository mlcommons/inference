#!/bin/bash
#SBATCH --job-name=minimal
#SBATCH --partition=4n4gpu
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=144
#SBATCH -o tmp.out
#SBATCH -e tmp.out
#SBATCH --time=0:01:00


module load nvhpc

mpirun --map-by node:PE=144 nproc

echo "[DONE]"
