#!/bin/sh

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

# Assumptions:
# Equal user names and paths on both machines
# Assuming you are in PROJECT_DIR/remotelaunch and want to run SCRIPT_NAME.py
# Anaconda is installed at $HOME/Software/anaconda3/bin/conda
#
# YOU INSTALLED ACCORDING TO THE OPTION "Red Velvet", I.E. DO NOT USE RCS OR RCSPYSIM

# Usage:
# (remotelaunch_slurm_cpu.sh is set up such that the required script path starts at Pyrado/scripts)
# bash remotelaunch_slurm_cpu.sh training/SCRIPT_NAME.py

CMD="$@"

DSTHOST="shannon"
PROOT="$HOME/Software/SimuRLacra" # ADD PATH TO PROJECT ROOT DIR, I.E. PATH TO SimuRLacra

RLAUNCH_DIR="$PROOT/remotelaunch"

# The cluster node is taking care of activating the anaconda environment

# Synchronize code
$RLAUNCH_DIR/sync_to_host.sh $DSTHOST "$PROOT"

# Now, run all this on the remote host
ssh -t -t $DSTHOST << EOF
shopt -s expand_aliases

cd "$RLAUNCH_DIR"
sbatch slurm_launcher_cpu.sh $CMD

exit
EOF
