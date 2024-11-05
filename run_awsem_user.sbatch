#!/bin/bash
#SBATCH -t 47:59:00
#SBATCH -N 1
#SBATCH -A ABC123
#SBATCH -o awsem.output.%j
#SBATCH -e awsem.output.%j
#SBATCH -J awsem
source /etc/profile.d/modules.sh
module purge
export APPTAINER_CACHEDIR=/tahoma/$SLURM_JOB_ACCOUNT/$USER/cache
mkdir -p $APPTAINER_CACHEDIR
export https_proxy=http://proxy.emsl.pnl.gov:3128
export http_proxy=http://proxy.emsl.pnl.gov:3128
apptainer pull  -F /tmp/openmm.sif oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:latest
MYFS=$(findmnt -r -T . | tail -1 |cut -d ' ' -f 1)
export SHARED_DATA_DIR=/tahoma/scicons/datasets/openawsem/data
export DATA_DIR=/opt/conda/envs/py38/lib/python3.8/site-packages/openawsem/data
export MYDATA_DIR=$(pwd)/data
mkdir -p $MYDATA_DIR/Indices $MYDATA_DIR/Gros
export MYBINDS="--bind $MYFS"
export MYBINDS="$MYBINDS,$MYDATA_DIR:$DATA_DIR"
export MYBINDS="$MYBINDS,$SHARED_DATA_DIR/PDBs:$DATA_DIR/PDBs"
export MYBINDS="$MYBINDS,$MYDATA_DIR/Indices:$DATA_DIR/Indices,$MYDATA_DIR/Gros:$DATA_DIR/Gros"
echo MYBINDS is $MYBINDS
rsync -av $SHARED_DATA_DIR/pdb_seqres.txt $MYDATA_DIR/.
rsync -av $SHARED_DATA_DIR/notExistPDBsList $MYDATA_DIR/.
rsync -av $SHARED_DATA_DIR/database $MYDATA_DIR/.
apptainer exec  --bind $MYFS   /tmp/openmm.sif  python  -m openmm.testInstallation
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 8enr --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 1e67 --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 1e65 --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 1azu --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 3fpy --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 1jvl --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 1e5z --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 1e5y --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 1azn --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 7tc5 --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 2xv3 --frag
apptainer exec $MYBINDS oras://ghcr.io/edoapra/openmm-singularity/openmm-ubuntu_22.04:20241101_130716  awsem_create 2wsc --frag

