#!/bin/sh
#PBS -N Quench_e-4
#PBS -P civil
#PBS -q standard
#PBS -M $USER@iitd.ac.in
#PBS -m bea
#########################################
## Refer any of the below mentioned select statemnt
## as per the type of job you want to submit
## Keep single # before PBS to consider it as command ,
## more than one # before PBS considered as comment.
## any command/statement other than PBS starting with # is considered as comment.
## Please comment/uncomment the portion as per your requirement before submitting job


## CPU JOB
#PBS -l select=1:ncpus=4
#PBS -l walltime=00:40:00
#PBS -l software=LAMMPS

export OMP_NUM_THREADS=1

## Environment
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

module purge
module load apps/lammps/intel/7Aug19

## CPU JOB
mpirun -np $PBS_NTASKS lmp_mpi_cpu -in in.msd
