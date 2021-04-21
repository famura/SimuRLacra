# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os

import pyrado
from pyrado.utils.input_output import print_cbt


try:
    import mujoco_py
except (ImportError, Exception):
    # The ImportError is raised if mujoco-py is simply not installed
    # The Exception catches the case that you have everything installed properly but your IDE does not set the
    # LD_LIBRARY_PATH correctly (happens for PyCharm & CLion). To check this, try to run your script from the terminal.
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    ld_preload = os.environ.get("LD_PRELOAD")
    print_cbt(
        "You are trying to use are MuJoCo-based environment, but the required mujoco_py module can not be imported.\n"
        "Try adding\n"
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco200/bin\n"
        "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so\n"
        "to your shell's rc-file.\n"
        "The current values of the environment variables are:\n"
        f"LD_LIBRARY_PATH={ld_library_path}\n"
        f"LD_PRELOAD={ld_preload}"
        "If you are using PyCharm or CLion, also add the environment variables above to your run configurations. "
        "Note that the IDE will not resolve $USER for some reason, so enter the user name directly, "
        "or run it from your terminal.\n\n"
        "Here comes the mujoco-py error message:\n\n",
        "r",
    )
    pyrado.mujoco_loaded = False
else:
    pyrado.mujoco_loaded = True
