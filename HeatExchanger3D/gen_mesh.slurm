#!/bin/bash

#SBATCH --job-name=fluent_ascii_convert
#SBATCH --output=fluent_convert.out
#SBATCH --error=fluent_convert.err
#SBATCH --partition=cpu
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --account=bcnx-delta-cpu
#SBATCH --time=00:30:00

# Set ANSYS license and environment
export ANSYSLMD_LICENSE_FILE=1055@ansys.webstore.illinois.edu
export ANSYSI_SERVERS=2325@ansys.webstore.illinois.edu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/bbkg/Ansys/ansys_inc/v232/fluent/fluent23.2.0/addons/gpuapp/cromwell/amd/dependencies/lnamd64/q

# Run Fluent in batch mode with your journal file
/projects/bbkg/Ansys/ansys_inc/v232/fluent/bin/fluent 3ddp -g -i /projects/bcnx/kazumak2/MIMONet/HeatExchanger3D/test.jou
