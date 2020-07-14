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

from functools import update_wrapper


class Delegate:
    """
    Delegate to a class member object.
    
    Example:
    class Foo:
        def __init__(self, bar):
            self.bar = bar
        baz = Delegate("bar")
    foo = Foo()
    foo.baz <=> foo.bar.baz
    """

    def __init__(self, delegate, attr=None):
        self.delegate = delegate
        self.attr = attr

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(getattr(instance, self.delegate), self.attr)

    def __set__(self, instance, value):
        setattr(getattr(instance, self.delegate), self.attr, value)

    def __set_name__(self, owner, name):
        # Use outer name for inner name
        if self.attr is None:
            self.attr = name


class ReadOnlyDelegate:
    """
    Delegate to a class member object in a read-only way.
    
    Example:
    class Foo:
        def __init__(self, bar):
            self.bar = bar
        baz = Delegate("bar")
    foo = Foo()
    foo.baz <=> foo.bar.baz
    """

    def __init__(self, delegate, attr=None):
        self.delegate = delegate
        self.attr = attr

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(getattr(instance, self.delegate), self.attr)

    def __set_name__(self, owner, name):
        # Use outer name for inner name
        if self.attr is None:
            self.attr = name


class cached_property:
    """
    Decorator that turns a function into a cached property.
    When the property is first accessed, the function is called to compute the value. Later calls use the cached value.
    .. note:: Requires a `__dict__` field, so it won't work on named tuples.
    """

    def __init__(self, func):
        self._func = func
        self._name = func.__name__
        update_wrapper(self, func)

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Create it
        res = self._func(instance)
        # Store in dict, subsequent queries will use that value
        instance.__dict__[self._name] = res
        return res
