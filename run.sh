#!/bin/bash
apptainer exec --nv   ./openmm.simg  python  -m openmm.testInstallation
