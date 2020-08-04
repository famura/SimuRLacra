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
Script to test the algorithm in a sim-to-sim toy task
"""
import numpy as np

from pyrado.algorithms.cem import CEM
from pyrado.algorithms.reps import REPS
from pyrado.algorithms.sysid_as_rl import SysIdByEpisodicRL, DomainDistrParamPolicy
from pyrado.domain_randomization.domain_parameter import UniformDomainParam, NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper, DomainRandWrapperLive
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.dummy import DummyPolicy, IdlePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamSim.name, f'{SysIdByEpisodicRL.name}-{REPS.name}', seed=1001)

    # Environments
    env_hparams = dict(dt=1/100., max_steps=500)
    env_real = BallOnBeamSim(**env_hparams)
    env_real.domain_param = dict(
        # l_beam=2.2,
        ang_offset=5*np.pi/180
    )

    env_sim = BallOnBeamSim(**env_hparams)
    randomizer = DomainRandomizer(
        # NormalDomainParam(name='l_beam', mean=1.8, std=1e-3, clip_lo=1, clip_up=4),
        UniformDomainParam(name='ang_offset', mean=0*np.pi/180, halfspan=1*np.pi/180),
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)
    # dp_map = {0: ('l_beam', 'mean'), 1: ('l_beam', 'std')}
    dp_map = {0: ('ang_offset', 'mean'), 1: ('ang_offset', 'halfspan')}
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    # Policies
    behavior_policy = IdlePolicy(env_sim.spec)  # DummyPolicy(env_sim.spec)  #
    prior = DomainRandomizer(
        # NormalDomainParam(name='l_beam', mean=2, std=1e-3),
        UniformDomainParam(name='ang_offset', mean=5*np.pi/180, halfspan=1e-8*np.pi/180),
    )
    ddp_policy = DomainDistrParamPolicy(mapping=dp_map, prior=prior)

    # Algorithm
    # subrtn_hparam = dict(
    #     max_iter=100,
    #     eps=0.1,
    #     pop_size=10*ddp_policy.num_param,
    #     num_rollouts=1,
    #     expl_std_init=1.0,
    #     expl_std_min=0.02,
    #     num_epoch_dual=1000,
    #     grad_free_optim=False,
    #     lr_dual=5e-4,
    #     use_map=True,
    #     num_sampler_envs=6,
    # )
    # subrtn = REPS(ex_dir, env_sim, ddp_policy, **subrtn_hparam)
    subrtn_hparam = dict(
        max_iter=100,
        pop_size=40,
        num_rollouts=1,
        num_is_samples=4,
        expl_std_init=0.05,
        expl_std_min=0.001,
        extra_expl_std_init=0.,
        extra_expl_decay_iter=10,
        num_sampler_envs=6,
    )
    subrtn = CEM(ex_dir, env_sim, ddp_policy, **subrtn_hparam)

    algo_hparam = dict(
        metric=None,
        obs_dim_weight=[1., 1., 1., 1.],
        num_rollouts_per_distr=50,
        num_sampler_envs=subrtn_hparam['num_sampler_envs']
    )

    # Save the environments and the hyper-parameters (do it before the init routine of BayRn)
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(subrtrn=subrtn_hparam, subrtrn_name=REPS.name),
        dict(algo=algo_hparam, algo_name=SysIdByEpisodicRL.name, dp_map=dp_map)],
        ex_dir
    )

    algo = SysIdByEpisodicRL(subrtn, behavior_policy, **algo_hparam)

    # Jeeeha
    while algo.curr_iter < algo.max_iter and not algo.stopping_criterion_met():
        algo.logger.add_value(algo.iteration_key, algo.curr_iter)

        # Creat fake real-world data
        ro_real = []
        for _ in range(7):
            ro_real.append(rollout(env_real, behavior_policy, eval=True))

        algo.step(snapshot_mode='latest', meta_info=dict(rollouts_real=ro_real))

        algo.logger.record_step()

        algo._curr_iter += 1

    if algo.stopping_criterion_met():
        stopping_reason = 'Stopping criterion met!'
    else:
        stopping_reason = 'Maximum number of iterations reached!'

    if algo._policy is not None:
        print_cbt(f'{SysIdByEpisodicRL.name} finished training a {ddp_policy.name} '
                  f'with {ddp_policy.num_param} parameters. {stopping_reason}', 'g')
    else:
        print_cbt(f'{SysIdByEpisodicRL.name} finished training. {stopping_reason}', 'g')
