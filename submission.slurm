#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 1000mb
#SBATCH --ntasks 1

# list NVIDIA cards, make clean, load the cuda module, make, and run main
lspci -vvv |& grep "NVIDIA" |& tee slurm-lspci.out

make clean && \
  make main && \
  ./main
