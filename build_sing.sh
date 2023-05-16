#!/bin/bash
rm -f /tmp/openmm.simg
export https_proxy=http://proxy.emsl.pnl.gov:3128
export http_proxy=http://proxy.emsl.pnl.gov:3128
  apptainer build --nv --force --fakeroot /tmp/openmm.simg Singularity.def
rsync -av /tmp/openmm.simg .
