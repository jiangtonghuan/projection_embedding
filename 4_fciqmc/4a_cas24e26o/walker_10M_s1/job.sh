#!/bin/bash
#SBATCH -o log.out
#SBATCH --partition=C032M0256G
#SBATCH --qos=low
#SBATCH -J La2CuO4_neci
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=37:30:00

module load mkl/2017.1
module load intel/2018.1
module load impi/2018.1.163

srun /bin/hostname -s | sort -n >slurm.hosts

mpirun -n 128 -machinefile slurm.hosts /gpfs/share/home/1600011363/neci-20210708/build_3/bin/neci fciqmc_input > fciqmc_output
