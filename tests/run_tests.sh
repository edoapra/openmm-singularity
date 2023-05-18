#!/bin/bash
#wrapper to run openmm singularity image
apptainer exec --nv  --bind `pwd`  ../openmm.simg   python  02_lennard_jones.py
apptainer exec --nv  --bind `pwd`  ../openmm.simg   python  01_force_fields.py
apptainer exec --nv  --bind `pwd`  ../openmm.simg   python  01_noninteractive_notebook_on_hpc.py
