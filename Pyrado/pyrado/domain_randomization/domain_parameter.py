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

import torch as to
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from typing import Sequence, Union, List, Optional

import pyrado
import torch as to
from pyrado.utils.input_output import print_cbt
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


class DomainParam:
    """ Class to store and manage a (single) domain parameter a.k.a. physics parameter a.k.a. simulator parameter """

    def __init__(
        self,
        name: str,
        clip_lo: Optional[Union[int, float]] = -pyrado.inf,
        clip_up: Optional[Union[int, float]] = pyrado.inf,
        roundint: bool = False,
    ):
        """
        Constructor, also see the constructor of DomainRandomizer.

        :param name: name of the parameter
        :param clip_lo: lower value for clipping
        :param clip_up: upper value for clipping
        :param roundint: flags if the parameters should be rounded and converted to an integer
        """
        if not isinstance(name, str):
            raise pyrado.TypeErr(given=name, expected_type=str)

        self.name = name
        self.mean = None  # to be set by subclass
        self.clip_lo = clip_lo
        self.clip_up = clip_up
        self.roundint = roundint
        self.distr = None  # no randomization by default

    def __eq__(self, other):
        """ Check if two `DomainParam` are equal by comparing all attributes defined in `get_field_names()`. """
        if not isinstance(other, DomainParam):
            raise pyrado.TypeErr(given=other, expected_type=DomainParam)

        for fn in self.get_field_names():
            if getattr(self, fn) != getattr(other, fn):
                return False
        return True

    @staticmethod
    def get_field_names() -> List[str]:
        """ Get union of all hyper-parameters of all domain parameter distributions. """
        return ["name", "clip_lo", "clip_up", "roundint"]

    def adapt(self, domain_distr_param: str, domain_distr_param_value: Union[float, int, to.Tensor]):
        """
        Update this domain parameter.

        .. note::
            This function should by called by the subclasses' `adapt()` function.

        :param domain_distr_param: distribution parameter to update, e.g. mean or std
        :param domain_distr_param_value: new value of the distribution parameter
        """
        if domain_distr_param not in self.get_field_names():
            raise pyrado.KeyErr(
                msg=f"The domain parameter {self.name} does not have a domain distribution parameter "
                f"called {domain_distr_param}!"
            )
        setattr(self, domain_distr_param, domain_distr_param_value)

    def sample(self, num_samples: int = 1) -> List[to.Tensor]:
        """
        Generate new domain parameter values.

        :param num_samples: number of samples (sets of new parameter values)
        :return: list of tensors containing the new parameter values
        """
        if not isinstance(num_samples, int):
            raise pyrado.TypeErr(given=num_samples, expected_type=int)
        if num_samples <= 0:
            raise pyrado.ValueErr(given=num_samples, g_constraint="0")

        if self.distr is None:
            raise RuntimeError("Trying to sample a domain parameter without a specified distribution!")

        # Draw num_samples samples (rsample is not implemented for Bernoulli)
        sample_tensor = self.distr.sample(sample_shape=to.Size([num_samples]))

        # Clip the values
        sample_tensor = to.clamp(sample_tensor, self.clip_lo, self.clip_up)

        # Round values to integers if desired
        if self.roundint:
            sample_tensor = to.round(sample_tensor).to(to.int32)

        # Convert the large tensor into a list of small tensors
        return list(sample_tensor)


class UniformDomainParam(DomainParam):
    """ Domain parameter sampled from a normal distribution """

    def __init__(self, mean: Union[int, float, to.Tensor], halfspan: Union[int, float, to.Tensor], **kwargs):
        """
        Constructor

        :param mean: nominal parameter value
        :param halfspan: half interval
        :param kwargs: forwarded to `DomainParam` constructor
        """
        super().__init__(**kwargs)

        self.mean = mean
        self.halfspan = halfspan
        self.distr = Uniform(self.mean - self.halfspan, self.mean + self.halfspan, validate_args=True)

    @staticmethod
    def get_field_names() -> List[str]:
        return ["name", "mean", "halfspan", "clip_lo", "clip_up", "roundint"]

    def adapt(self, domain_distr_param: str, domain_distr_param_value: Union[float, int]):
        # Set the attributes
        super().adapt(domain_distr_param, domain_distr_param_value)

        # Re-create the distribution, otherwise the changes will have no effect
        try:
            self.distr = Uniform(self.mean - self.halfspan, self.mean + self.halfspan, validate_args=True)
        except ValueError as err:
            print_cbt(
                f"Inputs that lead to the ValueError from PyTorch Distributions:"
                f"\ndomain_distr_param = {domain_distr_param}\n"
                f"low = {self.mean - self.halfspan}\nhigh = {self.mean + self.halfspan}"
            )
            raise err


class NormalDomainParam(DomainParam):
    """ Domain parameter sampled from a normal distribution """

    def __init__(self, mean: Union[int, float, to.Tensor], std: Union[int, float, to.Tensor], **kwargs):
        """
        Constructor

        :param mean: nominal parameter value
        :param std: standard deviation
        :param kwargs: forwarded to `DomainParam` constructor
        """
        super().__init__(**kwargs)

        self.mean = mean
        self.std = std
        self.distr = Normal(self.mean, self.std, validate_args=True)

    @staticmethod
    def get_field_names() -> List[str]:
        return ["name", "mean", "std", "clip_lo", "clip_up", "roundint"]

    def adapt(self, domain_distr_param: str, domain_distr_param_value: Union[float, int]):
        # Set the attributes
        super().adapt(domain_distr_param, domain_distr_param_value)

        # Re-create the distribution, otherwise the changes will have no effect
        try:
            self.distr = Normal(self.mean, self.std, validate_args=True)
        except ValueError as err:
            print_cbt(
                f"Inputs that lead to the ValueError from PyTorch Distributions:"
                f"\ndomain_distr_param = {domain_distr_param}\nloc = {self.mean}\nscale = {self.std}"
            )
            raise err


class MultivariateNormalDomainParam(DomainParam):
    """ Domain parameter sampled from a normal distribution """

    def __init__(self, mean: Union[int, float, to.Tensor], cov: Union[list, to.Tensor], **kwargs):
        """
        Constructor

        :param mean: nominal parameter value
        :param cov: covariance
        :param kwargs: forwarded to `DomainParam` constructor
        """
        super().__init__(**kwargs)

        self.mean = to.as_tensor(mean, dtype=to.get_default_dtype()).view(-1)
        self.cov = to.as_tensor(cov, dtype=to.get_default_dtype())
        if not self.cov.ndim == 2:
            raise pyrado.ShapeErr(msg="The covariance needs to be given as a matrix!")
        self.distr = MultivariateNormal(self.mean, self.cov, validate_args=True)

    @staticmethod
    def get_field_names() -> List[str]:
        return ["name", "mean", "cov", "clip_lo", "clip_up", "roundint"]

    def adapt(self, domain_distr_param: str, domain_distr_param_value: to.Tensor):
        if domain_distr_param == "cov" and domain_distr_param_value < 0:
            raise pyrado.ValueErr(given_name="cov", ge_constraint="0")

        # Set the attributes
        super().adapt(domain_distr_param, domain_distr_param_value)

        # Re-create the distribution, otherwise the changes will have no effect
        try:
            self.distr = MultivariateNormal(self.mean, self.cov, validate_args=True)
        except ValueError as err:
            print_cbt(
                f"Inputs that lead to the ValueError from PyTorch Distributions:"
                f"\ndomain_distr_param = {domain_distr_param}\nloc = {self.mean}\ncov = {self.cov}"
            )
            raise err


class BernoulliDomainParam(DomainParam):
    """ Domain parameter sampled from a Bernoulli distribution """

    def __init__(self, val_0: Union[int, float], val_1: Union[int, float], prob_1: float, **kwargs):
        """
        Constructor

        :param val_0: value of event 0
        :param val_1: value of event 1
        :param prob_1: probability of event 1, equals 1 - probability of event 0
        :param kwargs: forwarded to `DomainParam` constructor
        """
        super().__init__(**kwargs)

        self.val_0 = val_0
        self.val_1 = val_1
        self.prob_1 = prob_1
        self.mean = val_0 * (1 - prob_1) + val_1 * prob_1
        self.distr = Bernoulli(self.prob_1, validate_args=True)

    @staticmethod
    def get_field_names() -> List[str]:
        return ["name", "mean", "val_0", "val_1", "prob_1", "clip_lo", "clip_up", "roundint"]

    def adapt(self, domain_distr_param: str, domain_distr_param_value: Union[float, int]):
        # Set the attributes
        super().adapt(domain_distr_param, domain_distr_param_value)

        # Re-create the distribution, otherwise the changes will have no effect
        try:
            self.distr = Bernoulli(self.prob_1, validate_args=True)
        except ValueError as err:
            print_cbt(
                f"Inputs that lead to the ValueError from PyTorch Distributions:"
                f"\ndomain_distr_param = {domain_distr_param}\nprobs = {self.prob_1}"
            )
            raise err

    def sample(self, num_samples: int = 1) -> List[to.Tensor]:
        """
        Generate new domain parameter values.

        :param num_samples: number of samples (sets of new parameter values)
        :return: list of tensors containing the new parameter values
        """
        if not isinstance(num_samples, int):
            raise pyrado.TypeErr(given=num_samples, expected_type=int)
        if num_samples <= 0:
            raise pyrado.ValueErr(given=num_samples, g_constraint="0")

        if self.distr is None:
            raise RuntimeError("Trying to sample a domain parameter without a specified distribution!")

        # Draw num_samples samples (rsample is not implemented for Bernoulli)
        sample_tensor = self.distr.sample(sample_shape=to.Size([num_samples]))

        # Sample_tensor contains either 0 or 1
        sample_tensor = (to.ones_like(sample_tensor) - sample_tensor) * self.val_0 + sample_tensor * self.val_1

        # Clip the values
        sample_tensor = to.clamp(sample_tensor, self.clip_lo, self.clip_up)

        # Round values to integers if desired
        if self.roundint:
            sample_tensor = to.round(sample_tensor).to(to.int32)

        # Convert the large tensor into a list of small tensors
        return list(sample_tensor)


class SelfPacedLearnerParameter(DomainParam):
    def __init__(
        self,
        name: str,
        target_mean: to.Tensor,
        target_cov_chol_flat: to.Tensor,
        context_mean: to.Tensor,
        context_cov_chol_flat: to.Tensor,
    ):
        assert target_mean.shape == context_mean.shape, "Target and context mean should have same shape!"
        assert (
            target_cov_chol_flat.shape == context_cov_chol_flat.shape
        ), "Target and context cov chols should have same shape!"

        super().__init__(name=name, mean=context_mean)

        self.target_mean = target_mean
        self.target_cov_chol_flat = target_cov_chol_flat
        self.context_mean = context_mean
        self.context_cov_chol_flat = context_cov_chol_flat

        self.dim = target_mean.shape[0]

        self._target_distribution = MultivariateNormal(self.target_mean, self.target_cov, validate_args=True)
        self._context_distribution = MultivariateNormal(self.context_mean, self.context_cov, validate_args=True)

        self._sample_buffer = None

    @staticmethod
    def get_field_names() -> Sequence[str]:
        return ["target_mean", "target_cov_chol_flat", "context_mean", "context_cov_chol_flat"]

    @property
    def target_distribution(self):
        return self._target_distribution

    @property
    def context_distribution(self):
        return self._context_distribution

    @property
    def target_cov(self):
        return to.diag(self.target_cov_chol_flat ** 2)

    @property
    def context_cov(self):
        return to.diag(self.context_cov_chol_flat ** 2)

    @property
    def sample_buffer(self):
        return self._sample_buffer

    def adapt(self, domain_distr_param: str, domain_distr_param_value: to.Tensor):
        # Set the attributes
        super().adapt(domain_distr_param, domain_distr_param_value)

        # Re-create the distributions, otherwise the changes will have no effect
        if domain_distr_param in ["target_mean", "target_cov_chol_flat"]:
            try:
                self._target_distribution = MultivariateNormal(self.target_mean, self.target_cov, validate_args=True)
            except ValueError as err:
                print_cbt(
                    f"Inputs that lead to the ValueError from PyTorch Distributions:\n"
                    f"target_distribution; domain_distr_param = {domain_distr_param}\nloc = {self.target_mean}\ncov = {self.target_cov}"
                )
                raise err
        if domain_distr_param in ["context_mean", "context_cov_chol_flat"]:
            try:
                self._context_distribution = MultivariateNormal(self.context_mean, self.context_cov, validate_args=True)
            except ValueError as err:
                print_cbt(
                    f"Inputs that lead to the ValueError from PyTorch Distributions:\n"
                    f"init_distribution; domain_distr_param = {domain_distr_param}\nloc = {self.context_mean}\ncov = {self.context_cov}"
                )
                raise err

    def sample(self, num_samples: int = 1) -> list:
        print(f"Sample ID: {id(self)}")
        assert isinstance(num_samples, int) and num_samples > 0
        samples = self._context_distribution.sample((num_samples,))
        if len(samples.shape) == 1:
            samples = samples.reshape((-1, 1))
        if self._sample_buffer is None:
            self._sample_buffer = samples
        else:
            self._sample_buffer = to.vstack([self._sample_buffer, samples])
        return list(samples)

    def reset(self):
        self._sample_buffer = None
