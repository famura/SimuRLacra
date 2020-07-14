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
Test MuJoCo-based Hopper environment with a random policy.
"""
from pyrado.environments.mujoco.openai_hopper import HopperSim
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.policies.dummy import DummyPolicy, IdlePolicy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Set up environment
    env = HopperSim()

    # Set up policy
    # policy = DummyPolicy(env.spec)
    policy = IdlePolicy(env.spec)

    # Simulate
    done, param, state = False, None, None
    while not done:
        env.reset()
        print_cbt(f'init obs (before): {env.observe(env.state)}', 'c')
        ro = rollout(env, policy, render_mode=RenderMode(text=False, video=True), eval=True,
                     reset_kwargs=dict(domain_param=param, init_state=env.state.copy()))
        print_domain_params(env.domain_param)
        print_cbt(f'init obs (after): {ro.observations[0]}', 'c')
        done, state, param = after_rollout_query(env, policy, ro)
