#! /bin/bash
# @ job_name = rohf
# @ job_type = MPICH
# @ error = job.err.$(jobid)
# @ output = job.out.$(jobid)
# @ class = sles15
# @ restart = no
# @ node = 1
# @ tasks_per_node = 40
# @ node_usage = not_shared
# @ wall_clock_limit = 999:59:00
# @ stack_limit = unlimited
# @ queue


basename=0b_mixedrhf


# . /home/software/Boost/1_55_0-intel/source.sh
# . /home/software/Boost/1_69_0-intel/source.sh
# module load ifort
# . /home/software/Boost/1_72_0-gcc92/source.sh
# . /usr/local/server/IntelStudio_2016/parallel_studio_xe_2016.1.056/psxevars.sh intel64
. ~/pyscf-dev.sh
. ~/src/libmsym/source_libmsym.sh
export NUM_THREADS=40
export OMP_NUM_THREADS=40

# export FI_PROVIDER=sockets
# export I_MPI_DAPL_PROVIDER=ofa-v2-ib0

export I_MPI_ROOT="OFF"
MPIPREFIX=""
# MPIPREFIX="/usr/local/server/IntelStudio_2015/impi/5.0.3.049/intel64/bin/mpiexec.hydra -bootstrap ll "
# export I_MPI_ROOT="/usr/local/server/IntelStudio_2015/impi/5.0.3.049"
# MPIPREFIX="mpirun -np 10"
SUFFIX=$USER/tmp/pyscf_${basename}_$LOADL_STEP_ID
#PLACE=/algpfs1
PLACE=/scratch
export TMPDIR=$PLACE/$SUFFIX
$MPIPREFIX printenv  | grep HOSTNAME
$MPIPREFIX rm -vr $PLACE/tmp/*
$MPIPREFIX mkdir -vp $TMPDIR



# WFDIR=./WF
# rsync -av $WFDIR/node0 $TMPDIR

# ALTMPDIR=/algpfs1/${SUFFIX}
# mkdir -p $ALTMPDIR
# mkdir -p $ALTMPDIR/node0
# LINKNAME=./bak_${basename}_$LOADL_STEP_ID
# rm ${LINKNAME}
# ln -s $ALTMPDIR ${LINKNAME}

ulimit -s unlimited
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_sequential.so
# export PYTHONPATH=/home/bogdanov/src/block2-preview/build_alxmp:${PYTHONPATH}
## run stuff here

python ${basename}.py > ${basename}.out 2>&1

# ## copy back DMRG WF and clean scratch
# rsync -av $TMPDIR/node0/{wave*,statefile*,Rotation*,StateInfo*,RestartReorder*,*.bin,*.txt,*.e}  $ALTMPDIR/node0
# rsync -av $TMPDIR/*  $ALTMPDIR/

# rm $TMPDIR/int/*hdf5
# rsync -av $TMPDIR/int  $ALTMPDIR

$MPIPREFIX rm -vr $TMPDIR
# mv ./int ./int_dryrun
