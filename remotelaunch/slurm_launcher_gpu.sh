#!/usr/bin/env bash

# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH and
# Technical University of Darmstadt. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. All advertising materials mentioning features or use of this software
#    must display the following acknowledgement: This product includes
#    software developed by the Honda Research Institute Europe GmbH.
# 4. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

###############################################################################
## SLURM Configurations
#SBATCH --job-name single_run_gpu
#SBATCH --array 0-0
#SBATCH --time 72:00:00
## Always leave ntasks value to 1. This is only used for MPI, which is not supported now.
#SBATCH --ntasks 1
## Specify the number of cores. The maximum is 32.
#SBATCH --cpus-per-task 8
## Leave this if you want to use a GPU per job. Remove it if you do not need it.
#SBATCH --gres=gpu:rtx2080:1
##SBATCH -C avx
#SBATCH --mem-per-cpu=2048
#SBATCH -o /home/muratore/Software/SimuRLacra/remotelaunch/logs/%A_%a-out.txt
#SBATCH -e /home/muratore/Software/SimuRLacra/remotelaunch/logs/%A_%a-err.txt
###############################################################################

# Your program call starts here
echo "Starting Job $SLURM_JOB_ID, Array Job $SLURM_ARRAY_JOB_ID Index $SLURM_ARRAY_TASK_ID"

# Activate the pyrado anaconda environment
eval "$($HOME/Software/anaconda3/bin/conda shell.bash hook)"
conda activate pyrado

# Move to scripts directory
SIMURLACRA_DIR="$HOME/Software/SimuRLacra"
SCRIPTS_DIR="$SIMURLACRA_DIR/Pyrado/scripts"
cd "$SCRIPTS_DIR"

# Run python scripts with provided command line arguments
CMD="$@" # all arguments for the script call starting from PATHTO/SimuRLacra/Pyrado/scripts (excluding "python")
python $CMD
