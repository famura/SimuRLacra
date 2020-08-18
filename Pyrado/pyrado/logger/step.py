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

import csv
import os
import os.path as osp
import torch as to
import numpy as np
from abc import ABC, abstractmethod
from contextlib import contextmanager
from tabulate import tabulate

import pyrado
from pyrado.logger import resolve_log_path
from torch.utils.tensorboard import SummaryWriter


class StepLogger:
    """
    Step-based progress logger.
    This class collects progress values during a step. At the end, the record_step function will pass the collected
    values to one or more StepLogPrinters.
    The logger also validates that no values  are added unexpectedly, which i.e. a csv printer would not support.
    """

    def __init__(self, print_interval: int = 1):
        """
        Constructor

        :param print_interval: interval size, by default the logger records and prints on every call, i.e. every step
        """
        # Printer list (starts empty)
        self.printers = []
        # Values for current step
        self._current_values = {}
        # Known value keys in order of appearance
        self._value_keys = []
        # Set to false once the first step was written, no new keys are allowed
        self._first_step = True
        # Track if add_value was called since the last record_step()
        self._values_changed = False

        # Prefix management
        self._prefix_stack = []
        self._prefix_str = ''

        # Internal interval and counter
        self.print_interval = int(print_interval)
        self._counter = 0

    # Value management
    def add_value(self, key: str, value):
        """
        Add a column value to the current step.

        :param key: data key
        :param value: value to record, pass '' to print nothing
        """
        # Compute full prefixed key
        key = self._prefix_str + key

        if self._first_step:
            # Record new key during first step
            self._value_keys.append(key)
        elif key not in self._value_keys:
            # Make sure the key was used during first step
            raise KeyError('New value keys may only be added before the first step is finished')

        # Pre-process non-scalar values
        if isinstance(value, to.Tensor):
            # Only support scalar tensor for now
            if value.ndimension() <= 1:
                value = value.item()
            else:
                raise pyrado.ShapeErr(msg='Logger only support scalar PyTorch Tensors, otherwise the progress.csv file'
                                          ' gets meed up.')

        # Record value
        self._current_values[key] = value
        self._values_changed = True

    def record_step(self):
        """
        Record the currently stored values as step and print them at the end.
        To properly support nesting, this method does nothing if called twice in a row without an add_value in between.
        """
        # Only record a step if it is different from the last one
        if self._values_changed:
            self._values_changed = False

            # Use copy of values in case the printer decides to keep the dict
            # (Only affects the mock in the tests right now, but you never know...)
            values = self._current_values.copy()

            # Print only once every print_interval calls
            if self._counter % self.print_interval == 0:
                # Pass values to printers
                for p in self.printers:
                    p.print_values(values, self._value_keys, self._first_step)

            # Definitely not the first step any more
            self._first_step = False

        # Increase call counter
        self._counter += 1

    # Prefix management
    def push_prefix(self, pfx):
        """
        Push a string onto the key prefix stack.

        :param pfx: prefix string
        """
        self._prefix_stack.append(pfx)
        self._prefix_str = ''.join(self._prefix_stack)

    def pop_prefix(self):
        """ Remove the last string from the key prefix stack. """
        self._prefix_stack.pop()
        self._prefix_str = ''.join(self._prefix_stack)

    @contextmanager
    def prefix(self, pfx: str):
        """
        Context manager to add a prefix to the key prefix stack during use.

        :param pfx: prefix string
        """
        self.push_prefix(pfx)
        yield
        self.pop_prefix()


def create_csv_step_logger(save_dir: str, file_name: str = 'progress.csv') -> StepLogger:
    """
    Create a step-based logger which only safes to a csv-file.

    :param save_dir: parent directory to save the results in (usually the algorithm's `save_dir`)
    :param file_name: name of the cvs-file (with ending)
    :return: step-based logger
    """
    logger = StepLogger()

    logfile = osp.join(save_dir, file_name)
    logger.printers.append(CSVPrinter(logfile))
    return logger


class StepLogPrinter(ABC):
    """ Base class for log printers. Formats progress values for a step. """

    @abstractmethod
    def print_values(self, values: dict, ordered_keys: list, first_step: bool):
        """
        Print the values for a step.

        :param values: named progress values
        :param ordered_keys: value keys in a consistent order
        :param first_step: `True` for the first recorded step
        """


class ConsolePrinter(StepLogPrinter):
    """ Prints step data to the console """

    def print_values(self, values: dict, ordered_keys: list, first_step: bool):
        # One column with name, one column with value
        tbl = tabulate(
            [(k, values[k]) for k in ordered_keys], tablefmt='simple'
        )
        print(tbl)


class CSVPrinter(StepLogPrinter):
    """ Logs step data to a CSV file """

    def __init__(self, file: str):
        """
        Constructor

        :param file: csv file name
        """
        file = resolve_log_path(file)

        # Make sure the directory exists
        os.makedirs(osp.dirname(file), exist_ok=True)

        # Open file descriptor
        self.file = file

        self._fd = open(file, 'w')
        self._writer = csv.writer(self._fd)

    def print_values(self, values: dict, ordered_keys: list, first_step: bool):
        # Print header in first step
        if first_step:
            self._writer.writerow(ordered_keys)

        # Print ordered cell values
        self._writer.writerow([values[k] for k in ordered_keys])

        # Make sure we update the disk
        self._fd.flush()

    # Only serialize file name
    def __getstate__(self):
        return {'file': self.file}

    # And reopen the file for append on reload
    def __setstate__(self, state):
        self.file = state['file']

        self._fd = open(self.file, 'a')
        self._writer = csv.writer(self._fd)


class TensorBoardPrinter(StepLogPrinter):
    def __init__(self, file):
        file = resolve_log_path(file)
        self.writer = SummaryWriter(log_dir=file)
        self.step = 0

    def print_values(self, values: dict, ordered_keys: list, first_step: bool):
        for k in ordered_keys:
            value = values[k]
            if isinstance(value, list):
                for i, scalar in enumerate(value):
                   self.writer.add_scalar(k + str(i), scalar, self.step)
            elif isinstance(value, np.ndarray):
                for i, scalar in enumerate(value.flat):
                   self.writer.add_scalar(k + '/' + str(i), scalar, self.step)
            else:
                self.writer.add_scalar(k, values[k], self.step)
        self.step += 1
        self.writer.flush()


class LoggerAware:
    """
    Base for objects holding a StepLogger.
    Features automatic detection of child LoggerAware objects. Override to customize.
    """

    _logger: StepLogger = None
    _save_dir: str = None  # set this in the constructor of subclasses like Algorithm

    def _create_default_logger(self) -> StepLogger:
        """ Create a step-based logger which safes to a csv-file and prints to the console. """
        logger = StepLogger()
        logger.printers.append(ConsolePrinter())

        logfile = 'progress.csv'
        if self._save_dir is not None:
            logfile = osp.join(self._save_dir, logfile)
        logger.printers.append(CSVPrinter(logfile))
        return logger

    @property
    def logger(self) -> StepLogger:
        """ Get or create the step logger to use for this object. """
        if self._logger is not None:
            # There is already a logger object, so we can return it
            pass
        else:
            # Try to delegate to parent
            if hasattr(self, '_logger_parent'):
                return self._logger_parent.logger
            # Create a new one
            self._logger = self._create_default_logger()
        return self._logger

    def __setattr__(self, key, value):
        if isinstance(value, LoggerAware):
            # Setup as child
            value.__dict__['_logger_parent'] = self

        super().__setattr__(key, value)
