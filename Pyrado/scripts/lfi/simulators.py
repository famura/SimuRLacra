import os
import torch as to

import pyrado
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.policies.features import FeatureStack, identity_feat, sin_feat
from pyrado.sampling.rollout import rollout


class OscillatorTrajectories:
    def __init__(self, dt=0.005, max_steps=200, out_concat=True):
        self.env = OneMassOscillatorSim(dt=dt, max_steps=max_steps)
        self.policy = IdlePolicy(self.env.spec)
        self.out_concat = out_concat

    def __call__(self, mu):
        ro = rollout(self.env, self.policy, eval=True, reset_kwargs=dict(domain_param=dict(k=mu[0], d=mu[1])))
        if not self.out_concat:
            return to.tensor(ro.observations).to(dtype=to.float32)
        else:
            return to.tensor(ro.observations).to(dtype=to.float32).view(-1, 1).squeeze()


class BallOnBeam:
    def __init__(self, name=None, file_ext=None, load_dir=None, dt=0.02, max_steps=300, train=False):
        env_hparams = dict(dt=dt, max_steps=max_steps)
        self.env = BallOnBeamSim(**env_hparams)
        policy_hparam = dict(
            feats=FeatureStack([identity_feat, sin_feat])
            # feats=FeatureStack([RBFFeat(num_feat_per_dim=7, bounds=env.obs_space.bounds, scale=None)])
        )
        self.policy = LinearPolicy(spec=self.env.spec, **policy_hparam)
        name = "policy" if name is None else name
        file_ext = "pt" if file_ext is None else file_ext
        load_dir = "2020-12-15_15-00-26" if load_dir is None else load_dir
        if not train:
            self.policy = pyrado.load(
                self.policy, name="policy", file_ext="pt", load_dir="../data/temp/bob/cem_lin/2020-12-15_15-00-26"
            )

    def __call__(self, mu):
        ro = rollout(self.env, self.policy, eval=True, reset_kwargs=dict(domain_param=dict(g=mu[0], m_ball=mu[1])))
        return to.tensor(ro.observations).view(-1, 1).squeeze().to(dtype=to.float32)[:10]
