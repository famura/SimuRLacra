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

# Run this script on the local machine.
# Two argument: 1) the target host; 2) the directory to sync

if [ "$#" -ne 2 ]
  then
    echo "Missing host or directory argument"
    exit
fi
# Store arguments
DSTHOST="$1"
SYNCDIR="$2"

SRC="$SYNCDIR/"
DST="$DSTHOST:$SYNCDIR/"

echo "$SRC"

# Use rsync
# Archive, compress, progress, ssh algo, delete removed
# Exclude git and SVN files
rsync -azPe ssh --delete \
    --exclude "Pyrado/data/time_series" \
    --exclude "Pyrado/data/temp" \
    --exclude "thirdParty/" \
    --exclude "build/" \
    --exclude ".git/" \
    --exclude ".svn/" \
    --exclude "__pycache__" \
    --exclude-from="$(git -C "$SRC" ls-files --exclude-standard -oi --directory > /tmp/excludes; echo /tmp/excludes)" \
    "$SRC" "$DST"
