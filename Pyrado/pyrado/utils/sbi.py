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

from operator import itemgetter
from typing import List, Union, ValuesView

import numpy as np

import pyrado
from pyrado.sampling.sbi_embeddings import (
    AllStepsEmbedding,
    BayesSimEmbedding,
    DeltaStepsEmbedding,
    DynamicTimeWarpingEmbedding,
    Embedding,
    LastStepEmbedding,
    RNNEmbedding,
)
from pyrado.sampling.sbi_rollout_sampler import RolloutSamplerForSBI
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec


def create_embedding(name: str, env_spec: EnvSpec, *args, **kwargs) -> Embedding:
    """
    Create an embedding to use with sbi.

    :param name: identifier of the embedding
    :param env_spec: environment specification
    :param args: positional arguments forwarded to the embedding's constructor
    :param kwargs: keyword arguments forwarded to the embedding's constructor
    :return: embedding instance
    """
    if name == LastStepEmbedding.name:
        embedding = LastStepEmbedding(env_spec, RolloutSamplerForSBI.get_dim_data(env_spec), *args, **kwargs)
    elif name == DeltaStepsEmbedding.name:
        embedding = DeltaStepsEmbedding(env_spec, RolloutSamplerForSBI.get_dim_data(env_spec), *args, **kwargs)
    elif name == BayesSimEmbedding.name:
        embedding = BayesSimEmbedding(env_spec, RolloutSamplerForSBI.get_dim_data(env_spec), *args, **kwargs)
    elif name == DynamicTimeWarpingEmbedding.name:
        embedding = DynamicTimeWarpingEmbedding(env_spec, RolloutSamplerForSBI.get_dim_data(env_spec), *args, **kwargs)
    elif name == RNNEmbedding.name:
        embedding = RNNEmbedding(env_spec, RolloutSamplerForSBI.get_dim_data(env_spec), *args, **kwargs)
    elif name == AllStepsEmbedding.name:
        embedding = AllStepsEmbedding(env_spec, RolloutSamplerForSBI.get_dim_data(env_spec), *args, **kwargs)
    else:
        raise pyrado.ValueErr(
            given_name=name,
            eq_constraint=f"{LastStepEmbedding.name}, {DeltaStepsEmbedding.name}, {BayesSimEmbedding.name}, "
            f"{DynamicTimeWarpingEmbedding.name}, or {RNNEmbedding.name}",
        )

    return embedding


def check_domain_params(
    rollouts: Union[List[StepSequence], StepSequence],
    domain_param_value: np.ndarray,
    domain_param_names: Union[List[str], ValuesView],
):
    """
    Verify if the domain parameters in the rollout are actually the ones commanded.

    :param rollouts: simulated rollouts or rollout segments
    :param domain_param_value: one set of domain parameters as commanded
    :param domain_param_names: names of the domain parameters to set, i.e. values of the domain parameter mapping
    """
    if isinstance(rollouts, StepSequence):
        rollouts = [rollouts]

    if not all(
        [
            np.allclose(
                np.asarray(itemgetter(*domain_param_names)(ro.rollout_info["domain_param"])), domain_param_value
            )
            for ro in rollouts
        ]
    ):
        raise pyrado.ValueErr(
            msg="The domain parameters after the rollouts are not identical to the ones commanded by the sbi!"
        )
