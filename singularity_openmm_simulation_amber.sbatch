#!/bin/bash
#SBATCH -t 0:59:00
#SBATCH -N 1
#SBATCH -A emsla60288
#SBATCH -p analysis 
#SBATCH -o openmm_singularity.output.%j
#SBATCH -e openmm_singularity.output.%j
#SBATCH -J omm
source /etc/profile.d/modules.sh
module purge
module load gcc
export https_proxy=http://proxy.emsl.pnl.gov:3128
export http_proxy=http://proxy.emsl.pnl.gov:3128
apptainer pull  -F /tmp/openmm.sif oras://ghcr.io/edoapra/openmm-singularity/openmm:latest
#cp $ORGPWD/input.prmtop .
#cp $ORGPWD/input.inpcrd .
#cp $ORGPWD/run_openmm_simulation_amber.py .
MYFS=$(findmnt -r -T . | tail -1 |cut -d ' ' -f 1)
apptainer exec    /tmp/openmm.sif  python  -m openmm.testInstallation
apptainer exec --bind $MYFS   /tmp/openmm.sif     python ./run_openmm_simulation_amber.py

