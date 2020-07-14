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

"""
Convert and export a Policy (inherits from PyTorch's Module class) to C++ via TorchScript tracing/scripting.
The converted policy is saved same directory where the original policy was loaded from.

.. seealso::
    [1] https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
    [2] https://pytorch.org/tutorials/advanced/cpp_export.html
"""
import os.path as osp

import pyrado
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    if not isinstance(args.save_name, str):
        raise pyrado.TypeErr(given=args.save_name, expected_type=str)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Load the policy (trained in simulation)
    env, policy, _ = load_experiment(ex_dir)

    # Use torch.jit.trace / torch.jit.script (the latter if recurrent) to generate a torch.jit.ScriptModule
    ts_module = policy.trace()  # can be evaluated like a regular PyTorch module

    # Serialize the script module to a file and save it in the same directory we loaded the policy from
    export_file = osp.join(ex_dir, args.save_name + '.zip')
    ts_module.save(export_file)  # former: .pth

    # Export the experiment config for C++
    if isinstance(env, RcsSim):
        env.save_config_xml(osp.join(ex_dir, 'exTT_export.xml'))

    print_cbt(f'Exported the loaded policy to {export_file}', 'g', bright=True)
