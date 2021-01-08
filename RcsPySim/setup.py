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

import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            # Build a cmake extension
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir, "-DPYTHON_EXECUTABLE=" + sys.executable]

            cfg = "Debug" if self.debug else "Release"
            build_args = ["--config", cfg]
            build_args = ["--target", ext.name]

            if platform.system() == "Windows":
                cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)]
                if sys.maxsize > 2 ** 32:
                    cmake_args += ["-A", "x64"]
                build_args += ["--", "/m"]
            else:
                cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
                build_args += ["--", "-j2"]

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
            subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)
        else:
            super().build_extension(ext)


setup(
    name="RcsPySim",
    version="0.2",
    description="Robotic simulation environments backed by Rcs",
    author="Fabio Muratore & Felix Treede",
    author_email="fabio.muratore@famura.net",
    license="BSD 3-clause",
    # Specify empty py_modules to exclude pkgConfig.py
    py_modules=[],
    packages=["rcsenv"],
    # Source directory
    package_dir={"": "src/python/"},
    # Include config files
    package_data={"rcsenv": ["config/**/*"]},
    include_package_data=True,
    # Build CMake extensions
    ext_modules=[CMakeExtension("_rcsenv")],
    cmdclass=dict(build_ext=CMakeBuild),
    # Dependencies
    install_requires=["numpy"],
)
