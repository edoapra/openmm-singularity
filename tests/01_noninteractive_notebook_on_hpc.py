#!/usr/bin/env python
# coding: utf-8

# # Run demanding OpenMM notebooks non-interactively on an HPC
# 
# **<span style="color:#A03;font-size:14pt">
# &#x270B; HANDS-ON! &#x1F528;
# </span>**
# 
# 1. If you have not used the HPC yet for other parts of the tutorial, copy the following files to a directory on the HPC:
#    - `01_noninteractive_notebook_on_hpc.ipynb`
#    - `alanine-dipeptide.pdb`
#    - `job_openmm_hpc_cpu.sh` (job script for generic HPC setup, may need tweaking)
#    - `job_openmm_vsc_cpu.sh` (works on VSC clusters)
#    - `job_openmm_vsc_gpu.sh` (works on VSC clusters with GPUs)
# 
# 1. When you use a job script (`job_...`) for another notebook, you should edit it and change `01_noninteractive_notebook_on_hpc.ipynb` to the name of your notebook.
#    You can also change the `SBATCH` arguments to suit your needs:
#   
#    - Number of CPU cores is set by `--cpus-per-task`.
#    - Number of GPU cores is set by `--gpus-per-task`. (Not more than 1)
#    - Maximum wall time is set by `--time`. (Your job will be killed in case your calculation does not stop in time.)
#    - Maximum memory usage is set by `--mem`. (Your job will be killed when it uses more. If you don't specify this, the default is very low.)
#   
#    The `sbatch` program supports many other options to control the execution of the job.
#    Consult the [`sbatch` documentation](https://slurm.schedmd.com/sbatch.html) for more details.
# 
# 1. After creating the job script, run `sbatch job_openmm_cpu.sh`, or `sbatch job_openmm_gpu.sh` if the cluster has GPU nodes.
#    This will put your job on the queue, and as soon as resources are available to run your job, it will be executed.
#    
#    For **VSC** users:
#    To debug a job script, it is recommended to test on the Slaking cluster first.
#    Once it is working as expected, you can switch to a production cluster.
#    
#    The default cluster is `victini` when submitting jobs.
#    If you want to use another cluster, you need to run the following command, before calling `sbatch`:
#    
#    ```bash
#    module swap cluster/slaking
#    ```
#    
#    To see the available clusters:
#    
#    ```bash
#    module av cluster/
#    ```
# 
# 1. You can check the status of your job with the `squeue` command
#    VSC users can also check their job queue on [login.hpc.ugent.be](https://login.hpc.ugent.be).
#    Note that in a virtual terminal, you only get to see your jobs running on the current cluster.
#    Also for this, you need switch to the right cluster, e.g. with `module swap cluster/...`.
# 
# 1. Once the job has completed, a copy of this notebook is created with all the outputs 
#    (`01_run_openmm_on_a_hpc.nbconvert.ipynb`), and also all other output files can be found in the same directory.
# 
# **Notes for the GPU job script:**
# 
# - The GPU jobscript sets a variable `OPENMM_DEFAULT_PLATFORM=CUDA`, which tells OpenMM to use GPUs.
# 
# - For VSC users:
#   - It only makes sense to submit `job_openmm_gpu.sh` on `joltik` or `accelgor` Tier-2 clusters.
# 
#   - On `joltik`, it is recommended to use 8 CPUs per GPU.
#     On `accelgor`, this becomes 12 CPUs per GPU.
# 
#   - The GPU speedup for this notebook (on Joltik) is about a factor 30.

# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'widget')


# In[ ]:


# Import all the modules we need.
from sys import stdout

from openmm import *
from openmm.app import *
from openmm.unit import *


# The following code was taken from [../02/02_alanine_dipeptide.ipynb](../02/02_alanine_dipeptide.ipynb), example 3.

# In[ ]:


pdb = PDBFile("alanine-dipeptide.pdb")
modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
modeller.addSolvent(forcefield, model="tip3p", padding=1 * nanometer)
print(modeller.topology)
# Write a PDB file to provide a topology of the solvated
# system to MDTraj below.
with open("init3.pdb", "w") as outfile:
    PDBFile.writeFile(modeller.topology, modeller.positions, outfile)

# The modeller builds a periodic box with the solute and solvent molecules.
# PME is the method to compute long-range electristatic interactions in
# periodic systems.
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, constraints=HBonds)
temperature = 300 * kelvin
pressure = 1 * bar
integrator = LangevinIntegrator(temperature, 1 / picosecond, 2 * femtoseconds)
system.addForce(MonteCarloBarostat(pressure, temperature))
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()
simulation.reporters.append(DCDReporter("traj3.dcd", 100))
simulation.reporters.append(
    StateDataReporter(stdout, 1000, step=True, temperature=True, elapsedTime=True)
)
simulation.reporters.append(
    StateDataReporter(
        "scalars3.csv",
        100,
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
    )
)
simulation.step(100000)

