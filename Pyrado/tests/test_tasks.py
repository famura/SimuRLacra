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

import functools
import numpy as np
import pytest

import pyrado
from pyrado.tasks.condition_only import ConditionOnlyTask
from pyrado.tasks.desired_space import DesSpaceTask
from pyrado.utils.data_types import EnvSpec
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode, BestStateFinalRewTask
from pyrado.tasks.sequential import SequentialTasks
from pyrado.tasks.utils import proximity_succeeded
from pyrado.tasks.desired_state import DesStateTask, RadiallySymmDesStateTask
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.reward_functions import CompoundRewFcn, CosOfOneEleRewFcn, MinusOnePerStepRewFcn, QuadrErrRewFcn, \
    ScaledExpQuadrErrRewFcn, RewFcn, PlusOnePerStepRewFcn


@pytest.fixture(scope='function')
def envspec_432():
    return EnvSpec(obs_space=BoxSpace(-1, 1, 4), act_space=BoxSpace(-1, 1, 2), state_space=BoxSpace(-1, 1, 3))


@pytest.mark.parametrize(
    'fcn_list, reset_args, reset_kwargs', [
        ([MinusOnePerStepRewFcn()], [None], [None]),
        ([CosOfOneEleRewFcn(0)], [None], [None]),
        ([QuadrErrRewFcn(np.eye(2), np.eye(1))], [None], [None]),
        ([MinusOnePerStepRewFcn(), QuadrErrRewFcn(Q=np.eye(2), R=np.eye(1))], [None, None], [None, None]),
    ],
    ids=['wo_args-wo_kwargs', 'w_args-wo_kwargs', 'w_args2-wo_kwargs', 'wo_args-w_kwargs'])
def test_combined_reward_function_step(fcn_list, reset_args, reset_kwargs):
    # Create combined reward function
    c = CompoundRewFcn(fcn_list)
    # Create example state and action error
    err_s, err_a = np.array([1., 2.]), np.array([3.])
    # Calculate combined reward
    rew = c(err_s, err_a)
    assert isinstance(rew, float)
    # Reset the reward functions
    c.reset(reset_args, reset_kwargs)


def test_modulated_rew_fcn():
    Q = np.eye(4)
    R = np.eye(2)
    s = np.array([1, 2, 3, 4])
    a = np.array([0, 0])

    # Modulo 2 for all selected states
    idcs = [0, 1, 3]
    rew_fcn = QuadrErrRewFcn(Q, R)
    task = RadiallySymmDesStateTask(EnvSpec(None, None, None), np.zeros(4), rew_fcn, idcs, 2)
    r = task.step_rew(s, a)
    assert r == -(1**2 + 3**2)

    # Different modulo factor for the selected states
    idcs = [1, 3]
    task = RadiallySymmDesStateTask(EnvSpec(None, None, None), np.zeros(4), rew_fcn, idcs, np.array([2, 3]))
    r = task.step_rew(s, a)
    assert r == -(1**2 + 3**2 + 1**2)


@pytest.mark.parametrize(
    'state_space, act_space', [
        (BoxSpace(-np.ones((7,)), np.ones((7,))), BoxSpace(-np.ones((3,)), np.ones((3,)))),
    ],
    ids=['box']
)
def test_rew_fcn_constructor(state_space, act_space):
    r_m1 = MinusOnePerStepRewFcn()
    r_quadr = QuadrErrRewFcn(Q=5*np.eye(4), R=2*np.eye(1))
    r_exp = ScaledExpQuadrErrRewFcn(Q=np.eye(7), R=np.eye(3), state_space=state_space, act_space=act_space)
    assert r_m1 is not None
    assert r_quadr is not None
    assert r_exp is not None


@pytest.mark.parametrize(
    'task_type', [
        'ParallelTasks',
        'SequentialTasks'
    ],
    ids=['parallel', 'sequential']
)
def test_composite_task_structure(envspec_432, task_type):
    state_des1 = np.zeros(3)
    state_des2 = -.5*np.ones(3)
    state_des3 = +.5*np.ones(3)
    rew_fcn = MinusOnePerStepRewFcn()
    t1 = FinalRewTask(DesStateTask(envspec_432, state_des1, rew_fcn), mode=FinalRewMode(always_positive=True),
                      factor=10)
    t2 = FinalRewTask(DesStateTask(envspec_432, state_des2, rew_fcn), mode=FinalRewMode(always_positive=True),
                      factor=10)
    t3 = FinalRewTask(DesStateTask(envspec_432, state_des3, rew_fcn), mode=FinalRewMode(always_positive=True),
                      factor=10)

    if task_type == 'ParallelTasks':
        ct = ParallelTasks([t1, t2, t3])
    elif task_type == 'SequentialTasks':
        ct = SequentialTasks([t1, t2, t3])
    else:
        raise NotImplementedError
    ct.reset(env_spec=envspec_432)

    assert len(ct) == 3
    assert ct.env_spec.obs_space == envspec_432.obs_space
    assert ct.env_spec.act_space == envspec_432.act_space
    assert ct.env_spec.state_space == envspec_432.state_space
    assert isinstance(ct.tasks[0].rew_fcn, RewFcn)
    assert isinstance(ct.tasks[1].rew_fcn, RewFcn)
    assert isinstance(ct.tasks[2].rew_fcn, RewFcn)

    if type == 'ParallelTasks':
        assert np.all(ct.state_des[0] == state_des1)
        assert np.all(ct.state_des[1] == state_des2)
        assert np.all(ct.state_des[2] == state_des3)
    elif type == 'SequentialTasks':
        assert np.all(ct.state_des == state_des1)


@pytest.mark.parametrize(
    'hold_rew_when_done', [
        True,
        False
    ],
    ids=['hold_rews', 'dont_hold_rews']
)
def test_parallel_task_function(envspec_432, hold_rew_when_done):
    # Create env spec and sub-tasks (state_space is necessary for the has_failed function)
    state_des1 = np.zeros(3)
    state_des2 = -.5*np.ones(3)
    state_des3 = +.5*np.ones(3)
    rew_fcn = MinusOnePerStepRewFcn()
    succ_fcn = functools.partial(proximity_succeeded, thold_dist=1e-6)  # necessary to stop a sub-task on success
    t1 = FinalRewTask(DesStateTask(envspec_432, state_des1, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)
    t2 = FinalRewTask(DesStateTask(envspec_432, state_des2, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)
    t3 = FinalRewTask(DesStateTask(envspec_432, state_des3, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)

    pt = FinalRewTask(ParallelTasks([t1, t2, t3], hold_rew_when_done),
                      mode=FinalRewMode(always_positive=True), factor=100)

    # Create artificial dynamics by hard-coding a sequence of states
    num_steps = 12
    fixed_traj = np.linspace(-.5, +.6, num_steps, endpoint=True)  # for the final step, all sub-tasks would be true
    r = [-pyrado.inf]*num_steps

    for i in range(num_steps):
        # Advance the artificial state
        state = fixed_traj[i]*np.ones(3)

        # Get all sub-tasks step rew, check if they are done, and if so
        r[i] = pt.step_rew(state, act=np.zeros(2), remaining_steps=11 - i)

        # Check if reaching the sub-goals is recognized
        if np.all(state == state_des1):
            assert pt.wrapped_task.tasks[0].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 7  # only true for this specific setup
            else:
                assert r[i] == 8  # only true for this specific setup
        if np.all(state == state_des2):
            assert pt.wrapped_task.tasks[1].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 7  # only true for this specific setup
            else:
                assert r[i] == 7  # only true for this specific setup
        if np.all(state == state_des3):
            assert pt.wrapped_task.tasks[2].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 7  # only true for this specific setup
            else:
                assert r[i] == 9  # only true for this specific setup

        if i < 10:
            # The combined task is not successful until all sub-tasks are successful
            assert not pt.has_succeeded(state)
        elif i == 10:
            # Should succeed on the second to last
            assert pt.has_succeeded(state)
            assert pt.final_rew(state, 0) == pytest.approx(100.)
        elif i == 11:
            # The very last step reward
            if hold_rew_when_done:
                assert r[i] == -3.
            else:
                assert r[i] == 0.
            assert pt.final_rew(state, 0) == pytest.approx(0.)  # only yield once


@pytest.mark.parametrize(
    'hold_rew_when_done', [
        True,
        False
    ],
    ids=['hold_rews', 'dont_hold_rews']
)
def test_sequential_task_function(envspec_432, hold_rew_when_done):
    # Create env spec and sub-tasks (state_space is necessary for the has_failed function)
    state_des1 = -.5*np.ones(3)
    state_des2 = np.zeros(3)
    state_des3 = +.5*np.ones(3)
    rew_fcn = MinusOnePerStepRewFcn()
    succ_fcn = functools.partial(proximity_succeeded, thold_dist=1e-6)  # necessary to stop a sub-task on success
    t1 = FinalRewTask(DesStateTask(envspec_432, state_des1, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)
    t2 = FinalRewTask(DesStateTask(envspec_432, state_des2, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)
    t3 = FinalRewTask(DesStateTask(envspec_432, state_des3, rew_fcn, succ_fcn),
                      mode=FinalRewMode(always_positive=True), factor=10)

    st = FinalRewTask(SequentialTasks([t1, t2, t3], 0, hold_rew_when_done),
                      mode=FinalRewMode(always_positive=True), factor=100)

    # Create artificial dynamics by hard-coding a sequence of states
    num_steps = 12
    fixed_traj = np.linspace(-.5, +.6, num_steps, endpoint=True)  # for the final step, all sub-tasks would be true
    r = [-pyrado.inf]*num_steps

    for i in range(num_steps):
        # Advance the artificial state
        state = fixed_traj[i]*np.ones(3)

        # Get all sub-tasks step rew, check if they are done, and if so
        r[i] = st.step_rew(state, act=np.zeros(2), remaining_steps=num_steps - (i + 1))

        # Check if reaching the sub-goals is recognized
        if np.all(state == state_des1):
            assert st.wrapped_task.tasks[0].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 9  # only true for this specific setup
            else:
                assert r[i] == 9  # only true for this specific setup
        if np.all(state == state_des2):
            assert st.wrapped_task.tasks[1].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 8  # only true for this specific setup
            else:
                assert r[i] == 9  # only true for this specific setup
        if np.all(state == state_des3):
            assert st.wrapped_task.tasks[2].has_succeeded(state)
            if hold_rew_when_done:
                assert r[i] == 7  # only true for this specific setup
            else:
                assert r[i] == 9  # only true for this specific setup

        if i < 10:
            # The combined task is not successful until all sub-tasks are successful
            assert not st.has_succeeded(state)
        elif i == 10:
            # Should succeed on the second to last
            assert st.has_succeeded(state)
            assert st.final_rew(state, 0) == pytest.approx(100.)
        elif i == 11:
            # The very last step reward
            if hold_rew_when_done:
                assert r[i] == -3.
            else:
                assert r[i] == 0.
            assert st.final_rew(state, 0) == pytest.approx(0.)  # only yield once


@pytest.mark.parametrize(
    'rew_fcn', [
        PlusOnePerStepRewFcn(),
        QuadrErrRewFcn(np.eye(3), np.eye(2))
    ],
    ids=['PlusOnePerStepRewFcn', 'QuadrErrRewFcn']
)
@pytest.mark.parametrize(
    'factor', [1., 42.],
    ids=['factor_1', 'factor_42']
)
def test_best_state_final_rew_task(envspec_432, rew_fcn, factor):
    # Create env spec and sub-tasks (state_space is necessary for the has_failed function)
    num_steps = 5
    state_des = np.array([0.05, 0.05, 0.05])
    task = BestStateFinalRewTask(DesStateTask(envspec_432, state_des, rew_fcn), max_steps=num_steps, factor=factor)
    task.reset(env_spec=envspec_432)

    # Create artificial dynamics by hard-coding a sequence of states
    fixed_traj = np.linspace(-.5, +.5, num_steps, endpoint=True)
    r = [-pyrado.inf]*num_steps

    for i in range(num_steps):
        # Advance the artificial state
        state = fixed_traj[i]*np.ones(3)
        r[i] = task.step_rew(state, act=np.zeros(2), remaining_steps=num_steps - (i + 1))

    last_state = fixed_traj[-1]*np.ones(3)
    final_rew = task.final_rew(last_state, remaining_steps=0)
    assert final_rew == pytest.approx(max(r)*num_steps*factor)
    assert task.best_rew == pytest.approx(max(r))


@pytest.mark.parametrize(
    'rew_fcn', [
        QuadrErrRewFcn(0.1*np.eye(3), np.eye(2))
    ],
    ids=['QuadrErrRewFcn']
)
def test_tracking_task(envspec_432, rew_fcn):
    # Create env spec and sub-tasks (state_space is necessary for the has_failed function)
    num_steps = 5
    state_init = envspec_432.state_space.bound_lo.copy()
    state_des = envspec_432.state_space.bound_lo.copy()
    task = DesStateTask(envspec_432, state_des, rew_fcn)
    task.reset(env_spec=envspec_432)

    # Create artificial dynamics by hard-coding a sequence of states
    fixed_traj = np.linspace(-.5, +.5, num_steps, endpoint=True)
    r = [-pyrado.inf]*num_steps

    for i in range(num_steps):
        # Advance the desired state, but keep the system state
        old_state_des_task = task.state_des.copy()
        state_des[:] = fixed_traj[i]*np.ones(3)
        r[i] = task.step_rew(state_init, act=np.zeros(2), remaining_steps=num_steps - (i + 1))

        if i > 0:
            assert all(task.state_des >= old_state_des_task)  # desired state is moving away
            assert r[i] <= r[i - 1]  # reward goes down


@pytest.mark.parametrize(
    'sub_tasks', [
        [DesStateTask(
            EnvSpec(obs_space=BoxSpace(-1, 1, 4), act_space=BoxSpace(-1, 1, 2), state_space=BoxSpace(-1, 1, 3)),
            np.array([0.05, 0.05, 0.05]), MinusOnePerStepRewFcn()),
            DesSpaceTask(
                EnvSpec(obs_space=BoxSpace(-1, 1, 4), act_space=BoxSpace(-1, 1, 2), state_space=BoxSpace(-1, 1, 3)),
                BoxSpace(-1., 1., shape=3), MinusOnePerStepRewFcn())
        ]
    ],
    ids=['des_state_and_des_space']
)
def test_set_goals_fo_composite_tasks(sub_tasks):
    # Check ParallelTasks
    pt = ParallelTasks(sub_tasks)
    pt.state_des = 1*[np.array([-0.2, 0.4, 0])]
    assert np.all(pt.state_des == np.array([-0.2, 0.4, 0]))
    pt.space_des = 1*[BoxSpace(-0.5, 2., shape=3)]
    assert pt.space_des[0] == BoxSpace(-0.5, 2., shape=3)  # pt.space_des is a list

    # Check SequentialTasks
    st = SequentialTasks(sub_tasks)
    st.state_des = np.array([-0.2, 0.4, 0])
    assert np.all(st.state_des == np.array([-0.2, 0.4, 0]))
    st.idx_curr = 1
    st.space_des = BoxSpace(-0.5, 2., shape=3)
    assert st.space_des == BoxSpace(-0.5, 2., shape=3)


@pytest.mark.parametrize(
    'condition_fcn', [lambda x: np.linalg.norm(x - np.array([0.5, 0.5, 0.5])) < 0.01]
)
@pytest.mark.parametrize(
    'is_success_condition', [True, False],
    ids=['isc_true', 'isc_false']
)
def test_condition_only_task(envspec_432, condition_fcn, is_success_condition):
    cot = ConditionOnlyTask(envspec_432, condition_fcn, is_success_condition)
    cot.reset(envspec_432)

    state = np.array([0., 0., 0.5])
    assert not cot.has_failed(state)
    assert not cot.has_succeeded(state)

    state = np.array([0.5, 0.5, 0.5])
    if cot.is_success_condition:
        assert not cot.has_failed(state)
        assert cot.has_succeeded(state)
    else:
        assert cot.has_failed(state)
        assert not cot.has_succeeded(state)
