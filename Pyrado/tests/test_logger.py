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

import numpy as np
import os.path as osp
import pytest
import shutil

from pyrado import TEMP_DIR
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml, load_dict_from_yaml
from pyrado.utils.order import get_immediate_subdirs


def test_experiment():
    ex_dir = setup_experiment("testenv", "testalgo", "testinfo", base_dir=TEMP_DIR)

    # Get the directory that should have been created by setup_experiment
    parent_dir = osp.join(ex_dir, "..")

    assert osp.exists(ex_dir)
    assert osp.isdir(ex_dir)

    # Get a list of all sub-directories (will be one since testenv is only used for this test)
    child_dirs = get_immediate_subdirs(parent_dir)
    assert len(child_dirs) > 0

    # Delete the created folder recursively
    shutil.rmtree(osp.join(TEMP_DIR, "testenv"), ignore_errors=True)  # also deletes read-only files


def test_save_and_laod_yaml():
    ex_dir = setup_experiment("testenv", "testalgo", "testinfo", base_dir=TEMP_DIR)

    # Save test data to YAML-file (ndarrays should be converted to lists)
    save_list_of_dicts_to_yaml([dict(a=1), dict(b=2.0), dict(c=np.array([1.0, 2.0]).tolist())], ex_dir, "testfile")

    data = load_dict_from_yaml(osp.join(ex_dir, "testfile.yaml"))
    assert isinstance(data, dict)

    # Delete the created folder recursively
    shutil.rmtree(osp.join(TEMP_DIR, "testenv"), ignore_errors=True)  # also deletes read-only files
