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
from copy import deepcopy
from tabulate import tabulate
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from typing import Callable, Optional, Mapping

import pyrado
from pyrado.domain_randomization.domain_parameter import (
    DomainParam,
    NormalDomainParam,
    UniformDomainParam,
    BernoulliDomainParam,
    MultivariateNormalDomainParam,
)
from pyrado.utils.tensor import deepcopy_or_clone


class DomainRandomizer:
    """ Class for executing the domain randomization """

    def __init__(self, *domain_params: DomainParam):
        """
        Constructor

        :param domain_params: list or tuple of `DomainParam` instances
        """
        self.domain_params = []
        self.add_domain_params(*domain_params)
        self._params_pert_dict = None  # dict of domain params which are a list of tensors with as may eles as samples
        self._params_pert_list = None  # list of domain param samples which are a dict with a key and one values

    def __str__(self):
        """ Create a string that yields a table-like result for print. """
        # Collect all keys a.k.a. headers
        headers = []
        dps = deepcopy(self.domain_params)
        for dp in dps:
            headers.extend(dp.get_field_names())
            if isinstance(dp, MultivariateNormalDomainParam):
                # Do not print `tensor[..]`
                dp.mean = dp.mean.numpy()
                dp.cov = dp.cov.numpy()

        # Manually order them. A set would reduce the duplicated, too but yield a random order.
        headers_ordered = ["name", "mean"]
        if "std" in headers:
            headers_ordered.append("std")
        if "cov" in headers:
            headers_ordered.append("cov")
        if "halfspan" in headers:
            headers_ordered.append("halfspan")
        if "val_0" in headers:
            headers_ordered.append("val_0")
        if "val_1" in headers:
            headers_ordered.append("val_1")
        if "prob_1" in headers:
            headers_ordered.append("prob_1")
        if "clip_lo" in headers:
            headers_ordered.append("clip_lo")
        if "clip_up" in headers:
            headers_ordered.append("clip_up")
        if "roundint" in headers:
            headers_ordered.append("roundint")

        # Create string
        return tabulate(
            [[getattr(dp, h, None) for h in headers_ordered] for dp in dps], headers=headers_ordered, tablefmt="simple"
        )

    def add_domain_params(self, *domain_params: DomainParam, mapping: Optional[Mapping[int, str]] = None):
        """
        Add an arbitrary number of domain parameters with their distributions to the randomizer.

        :param domain_params: list or tuple of `DomainParam` instances
        :param mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass, length).
                        This only sets the same and is intended to be used to guarantee the right number and order of
                        domain parameters in the randomizer.
        """
        for dp in domain_params:
            if not isinstance(dp, DomainParam):
                raise pyrado.TypeErr(given=dp, expected_type=DomainParam)
            self.domain_params.append(dp)

        if mapping is not None:
            # Sort according to the indices held by the keys
            sorted_mapping = dict(sorted(mapping.items()))
            for _, value in sorted_mapping.items():
                if not isinstance(value, str):
                    raise pyrado.TypeErr(given=value, expected_type=str)
                self.domain_params.append(DomainParam(name=value))

    def randomize(self, num_samples: int):
        """
        Draw random parameters from the associated distributions.
        Internally stores a dict with parameter names as dict-keys and the new parameters in list form as dict values.

        :param num_samples: number of samples to draw for each parameter
        """
        if not isinstance(num_samples, int):
            raise pyrado.TypeErr(given=num_samples, expected_type=int)
        if num_samples <= 0:
            raise pyrado.ValueErr(given=num_samples, g_constraint="0")

        # Generate domain parameter samples
        keys = [dp.name for dp in self.domain_params]
        values = [dp.sample(num_samples) for dp in self.domain_params]

        # Fill the internal storage containers
        self._params_pert_dict = dict(zip(keys, values))
        self._params_pert_list = []
        for i in range(num_samples):
            d = dict()
            for k, v in zip(keys, values):
                d[k] = v[i]
            self._params_pert_list.append(d)

    def get_params(self, num_samples: int = -1, fmt: str = "list", dtype: str = "numpy") -> [list, dict]:
        """
        Get the values in the data frame of the perturbed parameters.

        :param num_samples: number of samples to be extracted from the randomizer
        :param fmt: format (list of dicts or dict of lists) in which the params should be returned
        :param dtype: data type in which the params should be returned
        :return: dict of num_samples perturbed values per specified param or one dict of one perturbed
        """
        if not isinstance(num_samples, int):
            raise pyrado.TypeErr(given=num_samples, expected_type=int)
        if num_samples <= -2 or num_samples == 0:
            raise pyrado.ValueErr(msg="The number of samples needs to be -1 or a positive integer!")
        if not (fmt.lower() == "list" or fmt.lower() == "dict"):
            raise pyrado.ValueErr(given=fmt, eq_constraint="list or dict")
        if not (dtype.lower() == "numpy" or dtype.lower() == "torch"):
            raise pyrado.ValueErr(given=dtype, eq_constraint="numpy or torch")

        copy = None

        if num_samples == -1 and len(self._params_pert_list) > 1:
            # Return all samples that the randomizer holds
            if fmt == "list":
                # Return a list with all domain parameter sets
                copy = deepcopy_or_clone(self._params_pert_list)
                if dtype == "numpy":  # nothing to be done for torch
                    for i in range(len(copy)):
                        for k in copy[i].keys():
                            copy[i][k] = copy[i][k].detach().numpy()

            elif fmt == "dict":
                # Returns a dict (as many entries as parameters) with lists as values (as many entries as samples)
                copy = deepcopy_or_clone(self._params_pert_dict)
                if dtype == "numpy":  # nothing to be done for torch
                    for key in copy.keys():
                        copy[key] = [samples.detach().numpy() for samples in copy[key]]

        elif num_samples == 1 or len(self._params_pert_list) == 1:
            # If only one sample is wanted or the internal list just contains 1 element
            copy = deepcopy_or_clone(self._params_pert_list[0])
            if dtype == "numpy":  # nothing to be done for torch
                for k in copy.keys():
                    copy[k] = copy[k].detach().numpy()
            if fmt == "list":  # nothing to be done for dict
                copy = [copy]

        elif num_samples >= 1:
            # Return a subset of all samples that the randomizer holds
            if fmt == "list":
                copy = deepcopy_or_clone(self._params_pert_list[:num_samples])
                # Return a list with the fist num_samples domain parameter sets
                if dtype == "numpy":  # nothing to be done for torch
                    for i in range(num_samples):
                        for k in copy[i].keys():
                            copy[i][k] = copy[i][k].detach().numpy()

            elif fmt == "dict":
                # Return a dict with as many keys as perturbed params and num_samples values for each of them
                copy = dict()
                for key in self._params_pert_dict:
                    # Only select the fist num_samples elements of the list
                    copy[key] = self._params_pert_dict[key][:num_samples]
                    if dtype == "numpy":  # nothing to be done for torch
                        copy[key] = [p.detach().numpy() for p in copy[key]]

        if copy is None:
            raise RuntimeError(f"Something when wrong during get_params({num_samples}, {fmt}, {dtype})!")

        return copy

    def adapt_one_distr_param(
        self, domain_param_name: str, domain_distr_param: str, domain_distr_param_value: [float, int]
    ):
        """
        Update the randomizer's domain parameter distribution for one domain parameter.

        :param domain_param_name: name of the domain parameter which's distribution parameter should be updated
        :param domain_distr_param: distribution parameter to update, e.g. mean or std
        :param domain_distr_param_value: new value of the distribution parameter
        """
        for dp in self.domain_params:
            if dp.name == domain_param_name:
                if domain_distr_param in dp.get_field_names():
                    # Set the new value
                    if not isinstance(domain_distr_param_value, (int, float, bool)):
                        pyrado.TypeErr(given=domain_distr_param_value, expected_type=[int, float, bool])
                    dp.adapt(domain_distr_param, domain_distr_param_value)
                else:
                    raise pyrado.KeyErr(
                        msg=f"The domain parameter {dp.name} does not have a domain distribution parameter "
                        f"called {domain_distr_param}!"
                    )

    def rescale_distr_param(self, param: str, scale: float):
        """
        Rescale a parameter for all distributions.

        :param param: name of the parameter to change (e.g. std, or cov)
        :param scale: scaling factor
        """
        if not scale >= 0:
            raise pyrado.ValueErr(given=scale, ge_constraint="0")

        for dp in self.domain_params:
            if hasattr(dp, param):
                # Scale the param attribute of the domain parameters object
                setattr(dp, param, scale * getattr(dp, param))

            # Also scale the distribution (afterwards)
            if isinstance(dp, UniformDomainParam):
                dp.distr = Uniform(dp.mean - dp.halfspan, dp.mean + dp.halfspan)
            if isinstance(dp, NormalDomainParam):
                dp.distr = Normal(dp.mean, dp.std)
            if isinstance(dp, MultivariateNormalDomainParam):
                dp.distr = MultivariateNormal(dp.mean, dp.cov)
            if isinstance(dp, BernoulliDomainParam):
                dp.distr = Bernoulli(dp.prob_1)
