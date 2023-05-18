#!/bin/bash
#wrapper to run openmm singularity image
apptainer exec --nv  --bind `pwd`  ./openmm.simg  python  $1 $2 $3 $4
#apptainer exec --nv   ./openmm.simg  python  -m openmm.testInstallation
