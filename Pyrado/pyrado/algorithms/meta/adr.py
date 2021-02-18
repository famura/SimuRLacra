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

from typing import Optional, Sequence

import numpy as np
import pyrado
import torch as to
from init_args_serializer import Serializable
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.step_based.svpg import SVPG
from pyrado.domain_randomization.domain_parameter import DomainParam
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.feed_forward.fnn import FNNPolicy
from pyrado.sampling.parallel_evaluation import eval_domain_params
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.step_sequence import StepSequence
from pyrado.spaces.box import BoxSpace
from pyrado.utils.data_types import EnvSpec
from torch import nn as nn
from tqdm import tqdm


class ADR(Algorithm):
    """
    Active Domain Randomization (ADR)

    .. seealso::
        [1] B. Mehta, M. Diaz, F. Golemo, C.J. Pal, L. Paull, "Active Domain Randomization", arXiv, 2019
    """

    name: str = "adr"

    def __init__(
        self,
        save_dir: str,
        env: Env,
        subrtn: Algorithm,
        max_iter: int,
        svpg_particle_hparam: dict,
        num_svpg_particles: int,
        num_discriminator_epoch: int,
        batch_size: int,
        svpg_learning_rate: float = 3e-4,
        svpg_temperature: float = 10,
        svpg_evaluation_steps: int = 10,
        svpg_horizon: int = 50,
        svpg_kl_factor: float = 0.03,
        svpg_warmup: int = 0,
        svpg_serial: bool = False,
        num_workers: int = 4,
        num_trajs_per_config: int = 8,
        max_step_length: float = 0.05,
        randomized_params: Sequence[str] = None,
        logger: Optional[StepLogger] = None,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment to train in
        :param subrtn: algorithm which performs the policy / value-function optimization
        :param max_iter: maximum number of iterations
        :param svpg_particle_hparam: SVPG particle hyperparameters
        :param num_svpg_particles: number of SVPG particles
        :param num_discriminator_epoch: epochs in discriminator training
        :param batch_size: batch size for training
        :param svpg_learning_rate: SVPG particle optimizers' learning rate
        :param svpg_temperature: SVPG temperature coefficient (how strong is the influence of the particles on each other)
        :param svpg_evaluation_steps: how many configurations to sample between training
        :param svpg_horizon: how many steps until the particles are reset
        :param svpg_kl_factor: kl reward coefficient
        :param svpg_warmup: number of iterations without SVPG training in the beginning
        :param svpg_serial: serial mode (see SVPG)
        :param num_workers: number of environments for parallel sampling
        :param num_trajs_per_config: number of trajectories to sample from each config
        :param max_step_length: maximum change of physics parameters per step
        :param randomized_params: which parameters to randomize
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(subrtn, Algorithm):
            raise pyrado.TypeErr(given=subrtn, expected_type=Algorithm)
        if not isinstance(subrtn.policy, Policy):
            raise pyrado.TypeErr(given=subrtn.policy, expected_type=Policy)

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, subrtn.policy, logger)
        self.log_loss = True

        # Store the inputs
        self.env = env
        self._subrtn = subrtn
        self._subrtn.save_name = "subrtn"
        self.num_particles = num_svpg_particles
        self.num_discriminator_epoch = num_discriminator_epoch
        self.batch_size = batch_size
        self.num_trajs_per_config = num_trajs_per_config
        self.warm_up_time = svpg_warmup
        self.svpg_evaluation_steps = svpg_evaluation_steps
        self.svpg_temperature = svpg_temperature
        self.svpg_lr = svpg_learning_rate
        self.svpg_max_step_length = max_step_length
        self.svpg_horizon = svpg_horizon
        self.svpg_kl_factor = svpg_kl_factor

        self.pool = SamplerPool(num_workers)
        self.curr_time_step = 0

        # Get the number of params
        if isinstance(randomized_params, list) and len(randomized_params) == 0:
            randomized_params = inner_env(self.env).get_nominal_domain_param().keys()
        self.params = [DomainParam(param, 1) for param in randomized_params]
        self.num_params = len(self.params)

        # Initialize reward generator
        self.reward_generator = RewardGenerator(
            env.spec, self.batch_size, reward_multiplier=1, lr=1e-3, logger=self.logger
        )

        # Initialize logbook
        self.sim_instances_full_horizon = np.random.random_sample(
            (self.num_particles, self.svpg_horizon, self.svpg_evaluation_steps, self.num_params)
        )

        # Initialize SVPG
        self.svpg_wrapper = SVPGAdapter(
            env,
            self.params,
            subrtn.expl_strat,
            self.reward_generator,
            horizon=self.svpg_horizon,
            num_rollouts_per_config=self.num_trajs_per_config,
            num_workers=num_workers,
        )
        self.svpg = SVPG(
            save_dir,
            self.svpg_wrapper,
            svpg_particle_hparam,
            max_iter,
            self.num_particles,
            self.svpg_temperature,
            self.svpg_lr,
            self.svpg_horizon,
            serial=svpg_serial,
            num_workers=num_workers,
            logger=logger,
        )
        self.svpg.save_name = "subrtn_svpg"

    @property
    def sample_count(self) -> int:
        return self._subrtn.sample_count  # TODO @Robin: account for multiple particles

    def compute_params(self, sim_instances: to.Tensor, t: int):
        """
        Computes the parameters

        :param sim_instances: Physics configurations trajectory
        :param t: time step to chose
        :return: parameters at the time
        """
        nominal = self.svpg_wrapper.nominal_dict()
        keys = nominal.keys()
        assert len(keys) == sim_instances[t][0].shape[0]

        params = []
        for sim_instance in sim_instances[t]:
            d = dict()
            for i, k in enumerate(keys):
                d[k] = (sim_instance[i] + 0.5) * (nominal[k])
            params.append(d)

        return params

    def step(self, snapshot_mode: str, meta_info: dict = None, parallel: bool = True):
        rand_trajs = []
        ref_trajs = []
        ros = []
        visited = []
        for i in range(self.svpg.num_particles):
            done = False
            svpg_env = self.svpg_wrapper
            state = svpg_env.reset()
            states = []
            actions = []
            rewards = []
            infos = []
            rand_trajs_now = []
            if parallel:
                with to.no_grad():
                    for t in range(10):
                        action = (
                            self.svpg.expl_strats[i](to.as_tensor(state, dtype=to.get_default_dtype()))
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        state = svpg_env.lite_step(action)
                        states.append(state)
                        actions.append(action)
                    visited.append(states)
                    rewards, rand_trajs_now, ref_trajs_now = svpg_env.eval_states(states)
                    rand_trajs += rand_trajs_now
                    ref_trajs += ref_trajs_now
                    ros.append(StepSequence(observations=states, actions=actions, rewards=rewards))
            else:
                with to.no_grad():
                    while not done:
                        action = (
                            self.svpg.expl_strats[i](to.as_tensor(state, dtype=to.get_default_dtype()))
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        state, reward, done, info = svpg_env.step(action)
                        print(self.params.array_to_dict(state), " => ", reward)
                        states.append(state)
                        rewards.append(reward)
                        actions.append(action)
                        infos.append(info)
                        rand_trajs += info["rand"]
                        ref_trajs += info["ref"]
                    ros.append(StepSequence(observations=states, actions=actions, rewards=rewards))
            self.logger.add_value(f"SVPG_agent_{i}_mean_reward", np.mean(rewards))
            ros[i].torch(data_type=to.DoubleTensor)
            for rt in rand_trajs_now:
                rt.torch(data_type=to.double)
                rt.observations = rt.observations.double().detach()
                rt.actions = rt.actions.double().detach()
            self._subrtn.update(rand_trajs_now)

        # Logging
        rets = [ro.undiscounted_return() for ro in rand_trajs]
        ret_avg = np.mean(rets)
        ret_med = np.median(rets)
        ret_std = np.std(rets)
        self.logger.add_value("avg rollout len", np.mean([ro.length for ro in rand_trajs]))
        self.logger.add_value("avg return", ret_avg)
        self.logger.add_value("median return", ret_med)
        self.logger.add_value("std return", ret_std)

        # Flatten and combine all randomized and reference trajectories for discriminator
        flattened_randomized = StepSequence.concat(rand_trajs)
        flattened_randomized.torch(data_type=to.double)
        flattened_reference = StepSequence.concat(ref_trajs)
        flattened_reference.torch(data_type=to.double)
        self.reward_generator.train(flattened_reference, flattened_randomized, self.num_discriminator_epoch)
        pyrado.save(
            self.reward_generator.discriminator, "discriminator", "pt", self.save_dir, meta_info=dict(prefix="adr")
        )

        if self.curr_time_step > self.warm_up_time:
            # Update the particles
            # List of lists to comply with interface
            self.svpg.update(list(map(lambda x: [x], ros)))
        flattened_randomized.torch(data_type=to.double)
        flattened_randomized.observations = flattened_randomized.observations.double().detach()
        flattened_randomized.actions = flattened_randomized.actions.double().detach()

        # np.save(f'{self.save_dir}actions{self.curr_iter}', flattened_randomized.actions)
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)
        self._subrtn.make_snapshot(snapshot_mode="best", curr_avg_ret=float(ret_avg))
        self.curr_time_step += 1

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subrtn of another algorithm
            pyrado.save(self.env, "env", "pkl", self.save_dir, meta_info)
            self.svpg.save_snapshot(meta_info)
        else:
            raise pyrado.ValueErr(msg=f"{self.name} is not supposed be run as a subrtn!")


class SVPGAdapter(EnvWrapper, Serializable):
    """ Wrapper to encapsulate the domain parameter search as a reinforcement learning problem """

    def __init__(
        self,
        wrapped_env: Env,
        parameters: Sequence[DomainParam],
        inner_policy: Policy,
        discriminator,
        step_length: float = 0.01,
        horizon: int = 50,
        num_rollouts_per_config: int = 8,
        num_workers: int = 4,
    ):
        """
        Constructor

        :param wrapped_env: the environment to wrap
        :param parameters: which physics parameters should be randomized
        :param inner_policy: the policy to train the subrtn on
        :param discriminator: the discriminator to distinguish reference environments from randomized ones
        :param step_length: the step size
        :param horizon: an svpg horizon
        :param num_rollouts_per_config: number of trajectories to sample per physics configuration
        :param num_workers: number of environments for parallel sampling
        """
        Serializable._init(self, locals())

        EnvWrapper.__init__(self, wrapped_env)

        self.parameters: Sequence[DomainParam] = parameters
        self.pool = SamplerPool(num_workers)
        self.inner_policy = inner_policy
        self.svpg_state = None
        self.count = 0
        self.num_trajs = num_rollouts_per_config
        self.svpg_max_step_length = step_length
        self.discriminator = discriminator
        self.max_steps = 8
        self._adapter_obs_space = BoxSpace(-np.ones(len(parameters)), np.ones(len(parameters)))
        self._adapter_act_space = BoxSpace(-np.ones(len(parameters)), np.ones(len(parameters)))
        self.horizon = horizon
        self.horizon_count = 0

    @property
    def obs_space(self):
        return self._adapter_obs_space

    @property
    def act_space(self):
        return self._adapter_act_space

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        assert domain_param is None
        self.count = 0
        if init_state is None:
            self.svpg_state = np.random.random_sample(len(self.parameters))
        return self.svpg_state

    def step(self, act: np.ndarray) -> tuple:
        # Clip the action according to the maximum step length
        action = np.clip(act, -1, 1) * self.svpg_max_step_length

        # Perform step by moving into direction of action
        self.svpg_state = np.clip(self.svpg_state + action, 0, 1)
        param_norm = self.svpg_state + 0.5
        rand_eval_params = [self.array_to_dict(param_norm * self.nominal())] * self.num_trajs
        norm_eval_params = [self.nominal_dict()] * self.num_trajs
        rand = eval_domain_params(self.pool, self.wrapped_env, self.inner_policy, rand_eval_params)
        ref = eval_domain_params(self.pool, self.wrapped_env, self.inner_policy, norm_eval_params)
        rewards = [self.discriminator.get_reward(traj) for traj in rand]
        reward = np.mean(rewards)
        info = dict(rand=rand, ref=ref)
        if self.count >= self.max_steps - 1:
            done = True
        else:
            done = False
        self.count += 1
        self.horizon_count += 1
        if self.horizon_count >= self.horizon:
            self.horizon_count = 0
            self.svpg_state = np.random.random_sample(len(self.parameters))

        return self.svpg_state, reward, done, info

    def lite_step(self, act: np.ndarray):
        """
        Performs a step without the step interface.
        This allows for parallel computation of prior steps.

        :param act: the action to perform
        :return: the observation after the step
        """
        action = np.clip(act, -1, 1) * self.svpg_max_step_length
        self.svpg_state = np.clip(self.svpg_state + action, 0, 1)
        return self.svpg_state

    def eval_states(self, states: Sequence[np.ndarray]):
        """
        Evaluate the states.

        :param states: the states to evaluate
        :return: respective rewards and according trajectories
        """
        flatten = lambda l: [item for sublist in l for item in sublist]
        sstates = flatten([[self.array_to_dict((state + 0.5) * self.nominal())] * self.num_trajs for state in states])
        rand = eval_domain_params(self.pool, self.wrapped_env, self.inner_policy, sstates)
        ref = eval_domain_params(
            self.pool, self.wrapped_env, self.inner_policy, [self.nominal_dict()] * (self.num_trajs * len(states))
        )
        rewards = [self.discriminator.get_reward(traj) for traj in rand]
        rewards = [np.mean(rewards[i * self.num_trajs : (i + 1) * self.num_trajs]) for i in range(len(states))]
        return rewards, rand, ref

    def params(self):
        return [param.name for param in self.parameters]

    def nominal(self):
        return [inner_env(self.wrapped_env).get_nominal_domain_param()[k] for k in self.params()]

    def nominal_dict(self):
        return {k: inner_env(self.wrapped_env).get_nominal_domain_param()[k] for k in self.params()}

    def array_to_dict(self, arr):
        return {k: a for k, a in zip(self.params(), arr)}


class RewardGenerator:
    """
    Class for generating the discriminator rewards in ADR. Generates a reward using a trained discriminator network.
    """

    def __init__(
        self,
        env_spec: EnvSpec,
        batch_size: int,
        reward_multiplier: float,
        lr: float = 3e-3,
        logger: StepLogger = None,
        device: str = "cuda" if to.cuda.is_available() else "cpu",
    ):

        """
        Constructor

        :param env_spec: environment specification
        :param batch_size: batch size for each update step
        :param reward_multiplier: factor for the predicted probability
        :param lr: learning rate
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        self.device = device
        self.batch_size = batch_size
        self.reward_multiplier = reward_multiplier
        self.lr = lr
        spec = EnvSpec(
            obs_space=BoxSpace.cat([env_spec.obs_space, env_spec.obs_space, env_spec.act_space]),
            act_space=BoxSpace(bound_lo=[0], bound_up=[1]),
        )
        self.discriminator = FNNPolicy(spec=spec, hidden_nonlin=to.tanh, hidden_sizes=[62], output_nonlin=to.sigmoid)
        self.loss_fcn = nn.BCELoss()
        self.optimizer = to.optim.Adam(self.discriminator.parameters(), lr=lr, eps=1e-5)
        self.logger = logger

    def get_reward(self, traj: StepSequence):
        traj = convert_step_sequence(traj)
        with to.no_grad():
            reward = self.discriminator.forward(traj).cpu()
            return to.log(reward.mean()) * self.reward_multiplier

    def train(
        self, reference_trajectory: StepSequence, randomized_trajectory: StepSequence, num_epoch: int
    ) -> to.Tensor:

        reference_batch = reference_trajectory.split_shuffled_batches(self.batch_size)
        random_batch = randomized_trajectory.split_shuffled_batches(self.batch_size)

        loss = None
        for _ in tqdm(range(num_epoch), "Discriminator Epoch", num_epoch):
            try:
                reference_batch_now = convert_step_sequence(next(reference_batch))
                random_batch_now = convert_step_sequence(next(random_batch))
            except StopIteration:
                break
            if reference_batch_now.shape[0] < self.batch_size - 1 or random_batch_now.shape[0] < self.batch_size - 1:
                break
            random_results = self.discriminator(random_batch_now)
            reference_results = self.discriminator(reference_batch_now)
            self.optimizer.zero_grad()
            loss = self.loss_fcn(random_results, to.ones(self.batch_size - 1, 1)) + self.loss_fcn(
                reference_results, to.zeros(self.batch_size - 1, 1)
            )
            loss.backward()
            self.optimizer.step()

            # Logging
            if self.logger is not None:
                self.logger.add_value("discriminator_loss", loss)
        return loss


def convert_step_sequence(trajectory: StepSequence):
    """
    Converts a StepSequence to a Tensor which can be fed through a Network

    :param trajectory: A step sequence containing a trajectory
    :return: A Tensor containing the trajectory
    """
    assert isinstance(trajectory, StepSequence)
    trajectory.torch()
    state = trajectory.get_data_values("observations")[:-1].double()
    next_state = trajectory.get_data_values("observations")[1::].double()
    action = trajectory.get_data_values("actions").narrow(0, 0, next_state.shape[0]).double()
    trajectory = to.cat((state, next_state, action), 1).cpu().double()
    return trajectory
