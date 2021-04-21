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
Test model learning using PyTorch and the One Mass Oscillator setup.
"""
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorDomainParamEstimator, OneMassOscillatorSim
from pyrado.policies.special.dummy import DummyPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Set up environment
    dp_gt = dict(m=2.0, k=20.0, d=0.8)  # ground truth
    dp_init = dict(m=1.0, k=22.0, d=0.4)  # initial guess
    dt = 1 / 50.0
    env = OneMassOscillatorSim(dt=dt, max_steps=400)
    env.reset(domain_param=dp_gt)

    # Set up policy
    # policy = IdlePolicy(env.spec)
    policy = DummyPolicy(env.spec)

    # Sample
    sampler = ParallelRolloutSampler(env, policy, num_workers=4, min_rollouts=50, seed=1)
    ros = sampler.sample()

    # Create a model for learning the domain parameters
    model = OneMassOscillatorDomainParamEstimator(dt=dt, dp_init=dp_init, num_epoch=50, batch_size=10)

    model.update(ros)

    print_cbt(f"true domain param   : {dp_gt}", "g")
    print_cbt(f"initial domain param: {dp_init}", "y")
    print_cbt(f"learned domain param: {model.dp_est.detach().cpu().numpy()}", "c")
