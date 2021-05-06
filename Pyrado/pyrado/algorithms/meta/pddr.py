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
from copy import deepcopy
from typing import Any, List, Tuple

import numpy as np
import torch as to

import pyrado
from pyrado.algorithms.base import Algorithm, InterruptableAlgorithm
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.logger.experiment import Experiment, ask_for_experiment
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


class PDDR(InterruptableAlgorithm):
    """Policy Distillation with Domain Randomization (PDDR)"""

    name: str = "pddr"

    def __init__(
        self,
        save_dir: str,
        env: Env,
        policy: Policy,
        logger: StepLogger = None,
        device: str = "cpu",
        lr: float = 5e-4,
        std_init: float = 0.15,
        min_steps: int = 1500,
        num_epochs: int = 10,
        max_iter: int = 500,
        num_teachers: int = 8,
        num_cpu: int = 3,
        teacher_extra: dict = None,
        teacher_policy: Policy = None,
        teacher_algo: callable = None,
        teacher_algo_hparam: dict() = None,
        randomizer: DomainRandomizer = None,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        :param device: device to use for updating the policy (cpu or gpu)
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                    By default, the learning rate is constant.
        :param std_init: initial standard deviation on the actions for the exploration noise
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param num_epochs: number of epochs (how often we iterate over the same batch)
        :param max_iter: number of iterations (policy updates)
        :param num_teachers: number of teachers that are used for distillation
        :param num_cpu: number of cpu cores to use
        :param teacher_extra: extra dict from PDDRTeachers algo. If provided, teachers are loaded from there
        :param teacher_policy: policy to be updated (is duplicated for each teacher)
        :param teacher_algo: algorithm class to be used for training the teachers
        :param teacher_algo_hparam: hyperparams to be used for teacher_algo
        :param randomizer: randomizer for sampling the teacher domain parameters. If None, the default one for env is used
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(policy, Policy):
            raise pyrado.TypeErr(given=policy, expected_type=Policy)

        # Call Algorithm's constructor.
        super().__init__(
            num_checkpoints=1, init_checkpoint=-1, save_dir=save_dir, max_iter=max_iter, policy=policy, logger=logger
        )

        # Store the inputs
        self.min_steps = min_steps
        self.num_epochs = num_epochs
        self.num_teachers = num_teachers
        self.num_cpu = num_cpu
        self.device = device
        self.env_real = env

        self.teacher_policies = []
        self.teacher_envs = []
        self.teacher_expl_strats = []
        self.teacher_critics = []
        self.teacher_ex_dirs = []

        # Teachers
        if teacher_policy is not None and teacher_algo is not None and teacher_algo_hparam is not None:
            if not isinstance(teacher_policy, Policy):
                raise pyrado.TypeErr(given=teacher_policy, expected_type=Policy)
            if not issubclass(teacher_algo, Algorithm):
                raise pyrado.TypeErr(given=teacher_algo, expected_type=Algorithm)

            if randomizer is None:
                self.randomizer = create_default_randomizer(env)
            else:
                assert isinstance(randomizer, DomainRandomizer)
                self.randomizer = randomizer

            self.set_random_envs()

            # Prepare folders
            self.teacher_ex_dirs = [os.path.join(self.save_dir, f"teachers_{idx}") for idx in range(self.num_teachers)]
            for idx in range(self.num_teachers):
                os.makedirs(self.teacher_ex_dirs[idx], exist_ok=True)

            # Create teacher algos
            self.algos = [
                teacher_algo(
                    save_dir=self.teacher_ex_dirs[idx],
                    env=self.teacher_envs[idx],
                    policy=deepcopy(teacher_policy),
                    logger=None,
                    **deepcopy(teacher_algo_hparam),
                )
                for idx in range(self.num_teachers)
            ]
        elif teacher_extra is not None:
            self.unpack_teachers(teacher_extra)
            assert self.num_teachers == len(self.teacher_policies)
            self.reset_checkpoint()
        else:
            self.load_teachers()
            if self.num_teachers < len(self.teacher_policies):
                print(
                    f"You have loaded {len(self.teacher_policies)} teachers. Only the first {self.num_teachers} will be used!"
                )
                self.prune_teachers()
            assert self.num_teachers == len(self.teacher_policies)
            self.reset_checkpoint()

        # Student
        self._expl_strat = NormalActNoiseExplStrat(self._policy, std_init=std_init)
        self._policy = self._policy.to(self.device)
        self.optimizer = to.optim.Adam([{"params": self.policy.parameters()}], lr=lr)

        # Environments
        self.samplers = [
            ParallelRolloutSampler(
                self.teacher_envs[t],
                deepcopy(self._expl_strat),
                num_workers=int(self.num_cpu / self.num_teachers),
                min_steps=self.min_steps,
            )
            for t in range(self.num_teachers)
        ]

        self.teacher_weights = np.ones(self.num_teachers)

        # Distillation loss criterion
        self.criterion = to.nn.KLDivLoss(log_target=True, reduction="batchmean")

    @property
    def expl_strat(self) -> NormalActNoiseExplStrat:
        return self._expl_strat

    def step(self, snapshot_mode: str, meta_info: dict = None):
        """
        Performs a single iteration of the algorithm. This includes collecting the data, updating the parameters, and
        adding the metrics of interest to the logger. Does not update the `curr_iter` attribute.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new highscore)
        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        # Save snapshot to save the correct iteration count
        self.save_snapshot()

        if self.curr_checkpoint == -1:
            self.train_teachers(snapshot_mode, None)
            self.reached_checkpoint()  # setting counter to 0

        if self.curr_checkpoint == 0:
            # Sample observations
            ros, rets, all_lengths = self.sample()

            # Log current progress
            self.logger.add_value("max return", np.max(rets), 4)
            self.logger.add_value("median return", np.median(rets), 4)
            self.logger.add_value("avg return", np.mean(rets), 4)
            self.logger.add_value("min return", np.min(rets), 4)
            self.logger.add_value("std return", np.std(rets), 4)
            self.logger.add_value("std var", self.expl_strat.std.item(), 4)
            self.logger.add_value("avg rollout len", np.mean(all_lengths), 4)
            self.logger.add_value("num total samples", np.sum(all_lengths))

            # Save snapshot data
            self.make_snapshot(snapshot_mode, np.mean(rets), meta_info)

            # Update policy and value function
            self.update(rollouts=ros)

    def sample(self) -> Tuple[List[List[StepSequence]], np.array, np.array]:
        """
        Samples observations from several samplers.

        :return: list of rollouts per sampler, list of all returns, list of all rollout lengths
        """
        ros = []
        rets = []
        all_lengths = []
        for sampler in self.samplers:
            samples = sampler.sample()
            ros.append(samples)
            rets.extend([sample.undiscounted_return() for sample in samples])
            all_lengths.extend([sample.length for sample in samples])

        return ros, np.array(rets), np.array(all_lengths)

    def update(self, *args: Any, **kwargs: Any):
        """Update the policy's (and value functions') parameters based on the collected rollout data."""
        obss = []
        losses = []
        for t in range(self.num_teachers):
            concat_ros = StepSequence.concat(kwargs["rollouts"][t])
            concat_ros.torch(data_type=to.get_default_dtype())
            obss.append(concat_ros.get_data_values("observations")[: self.min_steps])

        # Train student
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()

            loss = 0
            for t_idx, teacher in enumerate(self.teacher_policies):
                s_dist = self.expl_strat.action_dist_at(self.policy(obss[t_idx]))
                s_act = s_dist.sample()
                t_dist = self.teacher_expl_strats[t_idx].action_dist_at(teacher(obss[t_idx]))

                l = self.teacher_weights[t_idx] * self.criterion(t_dist.log_prob(s_act), s_dist.log_prob(s_act))
                loss += l
                losses.append([t_idx, l.item()])
            print(f"Epoch {epoch} Loss: {loss.item()}")
            loss.backward()
            self.optimizer.step()

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        pyrado.save(self.policy, "policy.pt", self.save_dir)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self.env_real, "env.pkl", self.save_dir)

    def _train_teacher(self, idx: int, snapshot_mode: str = "latest", seed: int = None):
        """
        Wrapper for use of multiprocessing: Trains one teacher.

        :param idx: index of the teacher to be trained
        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new high-score)
        :param seed: seed value for the random number generators, pass `None` for no seeding
        """
        self.algos[idx].train(snapshot_mode=snapshot_mode, seed=seed)

    def set_random_envs(self):
        """Creates random environments of the given type."""
        self.randomizer.randomize(num_samples=self.num_teachers)
        params = self.randomizer.get_params(fmt="dict", dtype="numpy")

        for e in range(self.num_teachers):
            self.teacher_envs.append(deepcopy(self.env_real))
            print({key: value[e] for key, value in params.items()})
            self.teacher_envs[e].domain_param = {key: value[e] for key, value in params.items()}

    def train_teachers(self, snapshot_mode: str = "latest", seed: int = None):
        """
        Trains all teachers.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new high-score)
        :param seed: seed value for the random number generators, pass `None` for no seeding
        """
        for idx in range(self.num_teachers):
            print_cbt(f"Training teacher {idx + 1} of {self.num_teachers}... ", "c")
            self._train_teacher(idx, snapshot_mode, seed)

        self.teacher_policies = [a.policy for a in self.algos]
        self.teacher_expl_strats = [a.expl_strat for a in self.algos]
        self.teacher_critics = [a.critic for a in self.algos]

    def load_teachers(self):
        """Recursively load all teachers that can be found in the current experiment's directory."""
        # Get the experiment's directory to load from
        ex_dir = ask_for_experiment(max_display=10, env_name=self.env_real.name, perma=False)
        self.load_teacher_experiment(ex_dir)
        if len(self.teacher_policies) < self.num_teachers:
            print(
                f"You have loaded {len(self.teacher_policies)} teachers - load at least {self.num_teachers - len(self.teacher_policies)} more!"
            )
            self.load_teachers()

    def load_teacher_experiment(self, exp: Experiment):
        """
        Load teachers from PDDRTeachers experiment.

        :param exp: the teacher's experiment object
        """
        _, _, extra = load_experiment(exp)
        self.unpack_teachers(extra)

    def unpack_teachers(self, extra: dict):
        """
        Unpack teachers from PDDRTeachers experiment.

        :param extra: dict with teacher data
        """
        self.teacher_policies.extend(extra["teacher_policies"])
        self.teacher_envs.extend(extra["teacher_envs"])
        self.teacher_expl_strats.extend(extra["teacher_expl_strats"])
        self.teacher_critics.extend(extra["teacher_critics"])
        self.teacher_ex_dirs.extend(extra["teacher_ex_dirs"])

    def prune_teachers(self):
        """Prune teachers to only use the first num_teachers of them."""
        self.teacher_policies = self.teacher_policies[: self.num_teachers]
        self.teacher_envs = self.teacher_envs[: self.num_teachers]
        self.teacher_expl_strats = self.teacher_expl_strats[: self.num_teachers]
        self.teacher_critics = self.teacher_critics[: self.num_teachers]
        self.teacher_ex_dirs = self.teacher_ex_dirs[: self.num_teachers]
