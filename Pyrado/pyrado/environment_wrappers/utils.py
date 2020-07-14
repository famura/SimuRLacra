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
Some helpers for wrapped environment chains.
A 'chain' consists of an environment and multiple EnvWrappers wrapping it.
The real environment is always at the end of the chain.
The modifying methods in this file assume that all EnvWrapper subclasses use Serializable properly,
and that it's ctor takes the wrapped environment as first positional parameter.
"""
from pyrado.environment_wrappers.base import EnvWrapper


def all_envs(env):
    """
    Iterates over the environment chain.

    :param env: outermost environment of the chain
    :return: an iterable over the whole chain from outermost to innermost
    """
    yield env
    while isinstance(env, EnvWrapper):
        env = env.wrapped_env
        yield env


def typed_env(env, tp):
    """
    Locate the first element in the chain that is an instance of the given type.
    Returns `None` if not found.

    :param env: outermost environment of the chain
    :param tp: the environment type to find, see isinstance for possible values.
    :return: the first environment with the given type
    """
    for penv in all_envs(env):
        if isinstance(penv, tp):
            return penv
    return None


def attr_env(env, attr):
    """
    Locate the first element in the chain that has an attribute of the given name.
    Returns `None` if not found.

    :param env: outermost environment of the chain
    :param attr: attribute name to search
    :return: the first environment with the given attribute
    """
    for penv in all_envs(env):
        if hasattr(penv, attr):
            return penv
    return None


def attr_env_get(env, attr):
    """
    Locate the first element in the chain that has an attribute of the given name and return the value of the attribute.
    Returns `None` if not found.

    :param env: outermost environment of the chain
    :param attr: attribute name to search
    :return: the value of the given attribute, taken from the first environment with the given attribute
    """
    for penv in all_envs(env):
        try:
            return getattr(penv, attr)
        except AttributeError:
            # So it's not here. Go on.
            pass
    return None


def inner_env(env):
    """
    Returns the innermost (a.k.a. non-wrapper) environment.

    :param env: outermost environment of the chain
    :return: the innermost environment of the chain
    """
    while isinstance(env, EnvWrapper):
        env = env.wrapped_env
    return env


def _replace_wrapped_env(env, new_wrapped):
    """
    Create a copy of the given environment with `wrapped_env` replaced by `new_wrapped`.

    :param env: environment to be re-wrapped
    :param new_wrapped: new wrapper
    :return: re-wrapped copy of the environment
    """
    assert isinstance(env, EnvWrapper)
    # Use the copy function provided by Serializable
    return env.copy(wrapped_env=new_wrapped)


def _recurse_envstack_update(head_env, op):
    """
    Goes from the outer most environment to the inner most environment and performs an operation on each of them.

    :param head_env: outer most environment
    :param op: op(env) may return
               - a new env to indicate modification and termination
               - the same env to indicate termination without change
               - None to continue deeper
    :return: changed environment chain
    """
    # Execute op on head
    opres = op(head_env)
    if opres is not None:
        # Op changed it, so return changed
        return opres

    if isinstance(head_env, EnvWrapper):
        # Recurse deeper
        recres = _recurse_envstack_update(head_env.wrapped_env, op)
        # Rebuild current env if needed
        if recres is not head_env.wrapped_env:
            return _replace_wrapped_env(head_env, recres)
    # Reached end
    return head_env


def remove_env(stack, key_type):
    """
    Remove an `EnvWrapper` of the given type from the environment chain and return the modified chain.
    The original stack is unmodified, but untouched parts will be shared.
    If the key is not found, nothing will be done and the original chain is returned.

    :param stack: outermost environment of the chain
    :param key_type: type of environment to remove
    :return: the modified environment chain
    """
    assert issubclass(key_type, EnvWrapper)

    def _remove_op(env):
        if isinstance(env, key_type):
            return env.wrapped_env

    return _recurse_envstack_update(stack, _remove_op)


def insert_env_before(stack, key_type, insert_type, *args, **kwargs):
    """
    Add an EnvWrapper of the given type right before key_type to the environment chain and return the modified chain.
    The original stack is unmodified, but untouched parts will be shared. If the key is not found,
    nothing will be done and the original chain is returned.

    :param stack: outermost environment of the chain
    :param key_type: Type of environment to insert before. May be None to use the innermost environment
    :param insert_type: type of environment to insert
    :param args: additional args for insert_type's ctor
    :param kwargs: additional kwargs for insert_type's ctor
    :return: the modified environment chain
    """
    assert key_type is None or issubclass(key_type, EnvWrapper)
    assert issubclass(insert_type, EnvWrapper)

    def _insert_op(env):
        if not isinstance(env, EnvWrapper) if key_type is None else isinstance(env, key_type):
            return insert_type(env, *args, **kwargs)

    return _recurse_envstack_update(stack, _insert_op)
