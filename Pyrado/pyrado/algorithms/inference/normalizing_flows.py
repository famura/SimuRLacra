from typing import Optional, Callable, Type
import os.path as osp

import joblib
import pyrado
import torch as to
from torch import optim

from pyrado.logger.step import StepLogger, TensorBoardPrinter, LoggerAware
from torch.distributions import Distribution

import nflows
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

class NormalizingFlow(LoggerAware):
    """
    SBI-Wrapper.
    This class currently only works with posterior estimators and currently excludes
    likelihood- and density-ratio-estimators. This might be added later.
    Examplary file in '/pyrado/scripts/lfi/....py'
    """

    def __init__(
        self,
        save_dir: str,
        num_layers: int = 5,
        num_features: int = 2,
        base_dist: nflows.distributions.Distribution = StandardNormal(shape=[2]),
        optimizer: optim.Optimizer = optim.Adam,
        logger: Optional[StepLogger] = None,
        save_name: str = "algo",
    ):
        self._save_dir = save_dir
        self._num_layers = num_layers
        self._base_dist = base_dist
        self._curr_iter = 0
        self._save_name = save_name

        if logger is not None:
            self._logger = logger

        # sbi should use the same summary writer as this algo
        summary_writer = None
        for p in self.logger.printers:
            if isinstance(p, TensorBoardPrinter):
                summary_writer = p.writer

        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=num_features))
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_features,
                                                                  hidden_features=4))
        transform = CompositeTransform(transforms)
        self.flow = Flow(transform, base_dist)
        self._optimizer = optimizer(self.flow.parameters())


    def train(self, samples, snapshot_mode: str, meta_info: dict = None, num_iter=100):
        print()
        for i in range(num_iter):
            loss = self.step(samples, snapshot_mode, meta_info)
            print("\r[NFLOWS] Iteration ({}|{}); Loss: {}".format(i, num_iter, loss), end='')
        print("\n[NFLOWS] Training ended")
        return self.flow

    def step(self, samples, snapshot_mode: str, meta_info: dict = None):
        """
        Trains the posterior using SNPE using observed rollouts and the prior distribution

        """
        x = samples.clone().detach()
        self._optimizer.zero_grad()
        loss = -self.flow.log_prob(inputs=x).mean()
        loss.backward()
        self._optimizer.step()
        self.make_snapshot(snapshot_mode=snapshot_mode, meta_info=meta_info)
        return loss


    def evaluate(
        self,
        obs_traj: to.Tensor,
        num_samples: int = 1000,
        compute_quantity: dict = None,
    ):
        """
        Evaluates the posterior by calculating parameter samples given observed data, its log probability
        and the simulated trajectory.
        """
        pass

    def make_snapshot(self, snapshot_mode: str, meta_info: dict = None):
        """
        Make a snapshot of the training progress.
        This method is called from the subclasses and delegates to the custom method `save_snapshot()`.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new highscore)
        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        self.save_snapshot(meta_info)
        if snapshot_mode == "latest":
            self.save_snapshot(meta_info)
        else:
            raise pyrado.ValueErr(given=snapshot_mode, eq_constraint="'latest', 'best', or 'no'")

    def save_snapshot(self, meta_info: dict = None):
        pyrado.save(self.flow, "normalizing_flow", "pt", self._save_dir, meta_info, use_state_dict=False)
