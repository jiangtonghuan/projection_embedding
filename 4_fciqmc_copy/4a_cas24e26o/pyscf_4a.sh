#! /bin/bash
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -p C064M0256G
#SBATCH --qos=low
#SBATCH -J La2CuO4
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=32
#SBATCH --time=48:00:00

basename=4a_FCIDUMPgen

python ${basename}.py > ${basename}.out 2>&1

