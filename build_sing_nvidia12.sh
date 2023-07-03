#!/bin/bash
MYIMG=openmm_nvidia12.simg
rm -f /tmp/$MYIMG
export https_proxy=http://proxy.emsl.pnl.gov:3128
export http_proxy=http://proxy.emsl.pnl.gov:3128
  apptainer build --nv --force --fakeroot /tmp/$MYIMG Singularity_12.def
rsync -av /tmp/$MYIMG .
