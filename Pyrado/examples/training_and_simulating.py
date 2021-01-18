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
This file provides a very basic example of how to train a policy in SimuRLacra.
Have a look at `Pyrado/scripts/training` for more examples.
"""
import pyrado
from pyrado.algorithms.episodic.cem import CEM
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat, sin_feat
from pyrado.policies.feed_forward.linear import LinearPolicy


"""
First, we create an `Experiment`, which basically is a folder (by default in `Pyrado/data/temp`). The experiments are 
stored using the following scheme: <base_dir>/<env_name>/<algo_name>/<timestamp>--<extra_info>.
"""
ex_dir = setup_experiment(BallOnBeamSim.name, f"{CEM.name}_{LinearPolicy.name}")

"""
We can set a seed before creating the modules in oder to be sure that they get initialized identically, every time we
start this script. Note, that sampling with multiple workers is not deterministic, so the result can still vary, but
this then does not depend on the initialization of the networks' weights.
"""
pyrado.set_seed(seed=0, verbose=True)

"""
Next, we set up an environment and wrap it as desired.
"""
env_hparams = dict(dt=1 / 50.0, max_steps=300)
env = BallOnBeamSim(**env_hparams)
env = ActNormWrapper(env)

"""
Since all policy require information about the environments observation and action space, we create the policy after
the environment. Depending on the policy class, there are different parameters to set.
"""
policy_hparam = dict(feats=FeatureStack([identity_feat, sin_feat]))
policy = LinearPolicy(spec=env.spec, **policy_hparam)

"""
An algorithm at least needs a directory to save in, an environment, and a policy to do its job. Some also need are
value function or more exotic stuff that should be created beforehand.
"""
algo_hparam = dict(
    max_iter=100,
    pop_size=100,
    num_rollouts=12,
    num_is_samples=20,
    expl_std_init=0.5,
    expl_std_min=0.02,
    extra_expl_std_init=1.0,
    extra_expl_decay_iter=5,
    full_cov=True,
    symm_sampling=False,
    num_workers=8,
)
algo = CEM(ex_dir, env, policy, **algo_hparam)

"""
Save the hyper-parameters to a yaml-file is optional, but I highly recommend it, since it facilitates infecting
how we designed the experiment. By default, if is saved as hyperparams.yaml. 
"""
save_dicts_to_yaml(
    dict(env=env_hparams, seed=0),
    dict(policy=policy_hparam),
    dict(algo=algo_hparam, algo_name=algo.name),
    save_dir=ex_dir,
)

"""
Finally, we can start training. Here is another option to seed, which becomes more interesting, once we use continue
learning from a saved experiment. See `Pyrado/scripts/continue.py`. In most cases, the training stops after reaching
`max_iter`. We can cancel the procedure any time. Note, that every instance of `Algorithm` saves at every iteration.
However, there are different `snapshot_mode`s, which determine what is saved.
"""
algo.train(seed=1)

"""
After training, we can do to `Pyrado/scripts/simulation` and execute `python sim_policy.py`. Now we can select and
experiment, i.e. environment and policy, to simulate. We can enter a number or a complete path.
"""
