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
# You are at HRI
# You running this script from PATH_TO/SimuRLacra/remotelaunch

# Usage:
# bash remotelaunch_cmp180X.sh ID_NUMBER python PATH_TO/SimuRLacra/Pyrado/scripts/training/SCRIPT_NAME.py

# Former:
# cmake "$RCS_SRC_DIR" -DWRITE_PACKAGE_REGISTRY=ON -DUSE_VORTEX=ESSENTIALS -DVORTEX_ESSENTIALS_DIR=/hri/sit/LTS/External/Vortex/6.8.1/bionic64 -DUSE_BULLET=2.83_float
# instead of
# cmake "$RCS_SRC_DIR"

ID="$1"
shift # eat first argument
CMD="$@" # the remaining arguments are the script call (including command line arguments)

DSTHOST="cmp1804-0$ID"

LDISK="/hri/localdisk/$USER"
PROOT="$LDISK/Software/SimuRLacra"

RLAUNCH_DIR="$PROOT/remotelaunch"

RCS_SRC_DIR="$LDISK/Software/SimuRLacra/Rcs"
RCS_BUILD_DIR="$LDISK/Software/SimuRLacra/Rcs/build"

RCSPYSIM_SRC_DIR="$PROOT/RcsPySim"
RCSPYSIM_BUILD_DIR="$RCSPYSIM_SRC_DIR/build"

# Synchronize code
$RLAUNCH_DIR/sync_to_host.sh $DSTHOST "$RCS_SRC_DIR" 
$RLAUNCH_DIR/sync_to_host.sh $DSTHOST "$PROOT" 

# Now, run all this on the remote host
ssh -t -t $DSTHOST << EOF
shopt -s expand_aliases

mkdir -p "$RCS_BUILD_DIR"
cd "$RCS_BUILD_DIR"
cmake "$RCS_SRC_DIR"
make -j20

cd "$PROOT"
conda activate pyrado

mkdir -p "$RCSPYSIM_BUILD_DIR"
cd "$RCSPYSIM_BUILD_DIR"
cmake "$RCSPYSIM_SRC_DIR/build"
make -j20

$CMD

exit
EOF
