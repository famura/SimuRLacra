#! /usr/bin/env python

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
# 3. Neither the name of the Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt. nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY DAMRSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse
import errno
import os
import os.path as osp
import shutil
import subprocess as sp
import sys
import tarfile
import tempfile
import zipfile
from urllib.request import urlretrieve

import yaml

# Get the project's root directory
project_dir = osp.dirname(osp.abspath(__file__))

# Check if we are in CI
CI = "CI" in os.environ
# Make sure the git submodules are up to date, otherwise this script might break them
if not CI:
    sp.check_call(["git", "submodule", "update", "--init"], cwd=project_dir)

# Check if we are in HRI by looking for the SIT envionment variable
IN_HRI = "SIT" in os.environ


# ================== #
# PARSE ARGS EAGERLY #
# ================== #
# Allows to use them in the configuration

parser = argparse.ArgumentParser(description="Setup RcsPySim dev env")
parser.add_argument(
    "tasks",
    metavar="task",
    type=str,
    nargs="*",
    help="Subtasks to execute. Suggested tasks are `all` (includes every feature) or `no_rcs` (excludes Rcs and RcsPysim). To get a list of all availibe tasks, run `python setup_deps.py`.",
)
parser.add_argument("--vortex", dest="vortex", action="store_true", default=False, help="Use vortex physics engine")
parser.add_argument("--no_vortex", dest="vortex", action="store_false", help="Do not use vortex physics engine")
parser.add_argument("--cuda", dest="usecuda", action="store_true", default=False, help="Use CUDA for PyTorch")
parser.add_argument("--no_cuda", dest="usecuda", action="store_false", help="Do not use CUDA for PyTorch")
parser.add_argument("--headless", action="store_true", default=False, help="Build in headless mode")
parser.add_argument(
    "--local_torch",
    dest="uselibtorch",
    action="store_true",
    default=True,
    help="Use the local libtorch from the thirdParty directory for RcsPySim",
)
parser.add_argument(
    "--no_local_torch",
    dest="uselibtorch",
    action="store_false",
    help="Do not use the local libtorch from the thirdParty directory for RcsPySim",
)
parser.add_argument(
    "--pip_check", action="store_true", default=False, help="Run ´pip check´ after installing the dependencies"
)
parser.add_argument("-j", default=1, type=int, help="Number of make threads")

args = parser.parse_args()
# Check for help print later, when the tasks are defined

# ====== #
# CONFIG #
# ====== #

# Common directories
dependency_dir = osp.join(project_dir, "thirdParty")
resources_dir = osp.join(dependency_dir, "resources")

# Global cmake prefix path
cmake_prefix_path = [
    # Anaconda env root directory
    os.environ["CONDA_PREFIX"]
]

# Required packages
required_packages = [
    # "g++-4.8",  # necessary for Vortex
    "qt5-default",  # conda install -c dsdale24 qt5 _OR_ conda install -c anaconda qt  __OR__ HEADLESS BUILD
    # conda install -c dsdale24 qt5 _OR_ conda install -c anaconda qt  __OR__ HEADLESS BUILD
    "libqwt-qt5-dev",
    "libbullet-dev",  # conda install -c conda-forge bullet
    "libfreetype6-dev",  # conda install -c anaconda freetype
    "libxml2-dev",  # conda install -c anaconda libxml2
    "libglu1-mesa-dev",  # conda install -c anaconda libglu
    "freeglut3-dev",  # conda install -c anaconda freeglut
    "mesa-common-dev",  # conda install -c conda-forge mesalib
    # conda install -c conda-forge openscenegraph __OR__ HEADLESS BUILD
    "libopenscenegraph-dev",
    "openscenegraph",  # conda install -c conda-forge openscenegraph __OR__ HEADLESS BUILD
    "liblapack-dev",  # conda install -c conda-forge lapack
    "doxygen",  # necessary for building the Rcs documentation
    "python3-distutils",  # necessary for installing PyTorch
]
# using --headless: conda install -c conda-forge bullet freetype libglu freeglut mesalib lapack

required_packages_mujocopy = [
    "libglew-dev",
    "libosmesa6-dev",
]

env_vars = {
    # Global cmake prefix path
    "CMAKE_PREFIX_PATH": ":".join(cmake_prefix_path)
}

# Number of threads for make
make_parallelity = args.j

# WM5
# wm5_download_url = 'https://www.geometrictools.com/Downloads/WildMagic5p17.zip'  # deprecated
wm5_download_url = "https://github.com/zhouxs1023/WildMagic/archive/master.zip"
wm5_src_dir = osp.join(dependency_dir, "WildMagic5")

wm5_config = "ReleaseDynamic"
wm5_modules = ["LibCore", "LibMathematics"]

wm5_include_dir = osp.join(wm5_src_dir, "SDK/Include")
wm5_library_dir = osp.join(wm5_src_dir, "SDK/Library", wm5_config)

# Rcs
rcs_src_dir = osp.join(project_dir, "Rcs")
rcs_build_dir = osp.join(rcs_src_dir, "build")
rcs_cmake_vars = {
    "USE_BULLET": "2.83_float",
    "ENABLE_C++11": "ON",
    # Eigen is off for now until the dependency issues are fixed in Rcs.
    # Must specify include dir for Eigen 3.2
    # "EIGEN3_INCLUDE_DIR": eigen_include_dir,
    # "USE_EIGEN": "ON",
    "USE_WM5": "ON",  # for advanced collision models
    "WRITE_PACKAGE_REGISTRY": "ON",
}
# Optional headless mode
if args.headless:
    rcs_cmake_vars["HEADLESS_BUILD"] = "ON"

# Bullet
if IN_HRI:
    # Use bullet double from SIT
    # double causes errors due to soft bodies
    rcs_cmake_vars["USE_BULLET"] = "2.83_float"
else:
    # Bullet from apt-get package is in float mode
    rcs_cmake_vars["USE_BULLET"] = "2.83_float"

# Vortex
if args.vortex:
    rcs_cmake_vars["USE_VORTEX"] = "ESSENTIALS"
else:
    rcs_cmake_vars["USE_VORTEX"] = "OFF"

# WM5 collision library
if not IN_HRI:
    rcs_cmake_vars["WM5_INCLUDE_DIR"] = wm5_include_dir
    rcs_cmake_vars["WM5_LIBRARY_DIR"] = wm5_library_dir

# Kuka iiwa meshes
iiwa_repo_version = "1.2.5"
iiwa_url = f"https://github.com/IFL-CAMP/iiwa_stack/archive/{iiwa_repo_version}.tar.gz"

# Schunk SDH meshes
sdh_repo_version = "0.6.14"
sdh_url = f"https://github.com/ipa320/schunk_modular_robotics/archive/{sdh_repo_version}.tar.gz"

# Barrett WAM meshes (Pyrado)
wam_repo_version = "354c6e9"
wam_url = f"https://github.com/psclklnk/self-paced-rl/archive/{wam_repo_version}.tar.gz"

# PyTorch
# NOTE: Assumes that the current environment does NOT already contain PyTorch!
pytorch_version = "1.7.0"
pytorch_git_repo = "https://github.com/pytorch/pytorch.git"
pytorch_src_dir = osp.join(dependency_dir, "pytorch")

# RcsPySim
rcspysim_src_dir = osp.join(project_dir, "RcsPySim")
rcspysim_build_dir = osp.join(rcspysim_src_dir, "build")
uselibtorch = "ON" if args.uselibtorch else "OFF"
rcspysim_cmake_vars = {
    "PYBIND11_PYTHON_VERSION": "3.7",
    "SETUP_PYTHON_DEVEL": "ON",
    "Rcs_DIR": rcs_build_dir,
    "USE_LIBTORCH": uselibtorch,  # use the manually build PyTorch from thirdParty/pytorch
    # Do NOT set CMAKE_PREFIX_PATH here, it will get overridden later on.
}

# Pyrado
pyrado_dir = osp.join(project_dir, "Pyrado")

# Robcom & SL
cppsctp_git_repo = "https://git.ias.informatik.tu-darmstadt.de/robcom/cppsctp.git"
sl_git_repo = "https://git.ias.informatik.tu-darmstadt.de/sl/sl.git"
robcom_git_repo = "https://git.ias.informatik.tu-darmstadt.de/robcom-2/robcom-2.0.git"

ias_dir = osp.join(dependency_dir, "ias")
cppsctp_dir = osp.join(dependency_dir, "ias", "cppsctp")
sl_dir = osp.join(dependency_dir, "ias", "sl")
robcom_dir = osp.join(dependency_dir, "ias", "robcom")

cppsctp_cmake_vars = {"IAS_DIR": ias_dir}
cppsctpinstall_dir = osp.join(cppsctp_dir, "install")
sl_cmake_vars = {"IAS_DIR": ias_dir, "BUILD_barrett": "ON"}

# ======= #
# HELPERS #
# ======= #


def mkdir_p(path):
    """ Create directory and parents if it doesn't exist. """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def downloadAndExtract(url, destdir, archiveContentPath=None):
    """ Download an archive and extract it to the given destination. """
    # Select archive format
    if url.endswith(".tar.gz"):
        suffix = ".tar.gz"
        path_attr = "path"
    elif url.endswith(".zip"):
        suffix = ".zip"
        path_attr = "filename"
    else:
        raise ValueError("Unsupported archive file: {}".format(url))

    if osp.exists(destdir) and len(os.listdir(destdir)) != 0:
        # Exists, skip
        return
    with tempfile.NamedTemporaryFile(suffix=suffix) as tf:
        print("Downloading {}...".format(url))
        urlretrieve(url, tf.name)

        print("Extracting {}...".format(url))

        # Ensure destdir exists
        mkdir_p(destdir)

        # Filter
        if archiveContentPath is not None:
            # We only want to extract one subdirectory
            # Taken from https://stackoverflow.com/a/43094365
            def members(ml):
                subfolder = osp.normpath(archiveContentPath)
                l = len(subfolder)
                for member in ml:
                    # Skip directories in zip
                    isdir = getattr(member, "is_dir", None)
                    if isdir and isdir():
                        continue

                    # Modify output path
                    path = getattr(member, path_attr)
                    rp = osp.relpath(path, subfolder)
                    if not rp.startswith(".."):
                        setattr(member, path_attr, rp)
                        yield member

        else:

            def members(ml):
                return ml

        if suffix == ".tar.gz":
            with tarfile.open(tf.name) as tar:
                tar.extractall(members=members(tar.getmembers()), path=destdir)
        else:
            with zipfile.ZipFile(tf.name) as zip:
                zip.extractall(members=members(zip.infolist()), path=destdir)


def buildCMakeProject(srcDir, buildDir, cmakeVars=None, env=env_vars, install_dir=None):
    """
    cd buildDir
    cmake srcDir -D...
    make
    (make install)
    """
    # Ensure build dir exists
    mkdir_p(buildDir)

    if env is not None:
        fullenv = dict(os.environ)
        fullenv.update(env)
        env = fullenv

    # Execute CMake command
    cmake_cmd = ["cmake", osp.relpath(srcDir, buildDir)]
    if cmakeVars is not None:
        for k, v in cmakeVars.items():
            if v is True:
                vstr = "ON"
            elif v is False:
                vstr = "OFF"
            else:
                vstr = v
            cmake_cmd.append("-D{}={}".format(k, vstr))
    if install_dir is not None:
        cmake_cmd.append("-DCMAKE_INSTALL_PREFIX={}".format(install_dir))
    sp.check_call(cmake_cmd, cwd=buildDir, env=env)

    # Execute make (build) command
    make_cmd = ["make", "-j{}".format(make_parallelity)]
    sp.check_call(make_cmd, cwd=buildDir, env=env)

    # Execute install command if desired
    if install_dir is not None:
        mkdir_p(install_dir)
        sp.check_call(["make", "install"], cwd=buildDir)


# =========== #
# SETUP TASKS #
# =========== #


def setup_dep_libraries():
    # Update
    quiet = [] if not CI else ["-qq"]
    sp.check_call(["sudo", "apt-get"] + quiet + ["update", "-y"])
    # Install dependencies
    sp.check_call(["sudo", "apt-get"] + quiet + ["install", "-y"] + required_packages + required_packages_mujocopy)


def setup_wm5():
    # Download the sources
    downloadAndExtract(wm5_download_url, wm5_src_dir, "WildMagic-master")

    # Build relevant modules
    for module in wm5_modules:
        sp.check_call(
            [
                "make",
                "-f",
                "makefile.wm5",
                "build",
                "CFG={}".format(wm5_config),
                "-j{}".format(make_parallelity),
            ],
            cwd=osp.join(wm5_src_dir, module),
        )


def setup_rcs():
    # Build Rcs. We already have it in the submodule
    buildCMakeProject(rcs_src_dir, rcs_build_dir, cmakeVars=rcs_cmake_vars)


def setup_pytorch():
    # Get PyTorch from git
    if not osp.exists(pytorch_src_dir):
        mkdir_p(pytorch_src_dir)
        sp.check_call(
            [
                "git",
                "clone",
                "--recursive",
                "--branch",
                "v{}".format(pytorch_version),
                pytorch_git_repo,
                pytorch_src_dir,
            ]
        )
    # Let it's setup do the magic
    env = os.environ.copy()
    env.update(env_vars)
    # CUDA is disabled by default
    env["USE_CUDA"] = "1" if args.usecuda else "0"
    # CUDA is disabled by default
    env["USE_CUDNN"] = "1" if args.usecuda else "0"
    # disable MKLDNN; mkl/blas deprecated error https://github.com/pytorch/pytorch/issues/17874
    env["USE_MKLDNN"] = "0"
    env["_GLIBCXX_USE_CXX11_ABI"] = "1"
    sp.check_call([sys.executable, "setup.py", "install"], cwd=pytorch_src_dir, env=env)


def setup_rcspysim():
    # Take care of RcsPySim
    buildCMakeProject(rcspysim_src_dir, rcspysim_build_dir, cmakeVars=rcspysim_cmake_vars)


def setup_iiwa():
    # The Kuka iiwa meshes
    downloadAndExtract(
        iiwa_url, osp.join(resources_dir, "iiwa_description"), f"iiwa_stack-{iiwa_repo_version}/iiwa_description"
    )

    # Copy the relevant mesh files into RcsPySim's config folder
    # We already have the .tri meshes in there, just gives them company.
    src_dir = osp.join(resources_dir, "iiwa_description/meshes/iiwa14")
    dst_dir = osp.join(rcspysim_src_dir, "config/iiwa_description/meshes/iiwa14")

    # Collision and visual for links 0 - 7
    print("Copying the KUKA iiwa meshes to the RcsPySim config dir ...")
    for catdir in ["collision", "visual"]:
        for lnum in range(8):
            fname = osp.join(catdir, f"link_{lnum}.stl")

            mkdir_p(osp.dirname(osp.join(dst_dir, fname)))
            shutil.copyfile(osp.join(src_dir, fname), osp.join(dst_dir, fname))
    print("Setting up the KUKA iiwa meshes is done.")


def setup_schunk():
    # The Schunk SDH meshes
    downloadAndExtract(
        sdh_url,
        osp.join(resources_dir, "schunk_description"),
        f"schunk_modular_robotics-{sdh_repo_version}/schunk_description",
    )

    # Copy the relevant mesh files into RcsPySim's config folder
    # We already have the .tri meshes in there, just gives them company.
    src_dir = osp.join(resources_dir, "schunk_description/meshes/sdh")
    dst_dir = osp.join(rcspysim_src_dir, "config/schunk_description/meshes/sdh")

    # Get all .stl files in the sdh subdir
    print("Copying the Schunk SDH meshes to the RcsPySim config dir ...")
    for fname in os.listdir(src_dir):
        if fname.endswith(".stl"):
            mkdir_p(osp.dirname(osp.join(dst_dir, fname)))
            shutil.copyfile(osp.join(src_dir, fname), osp.join(dst_dir, fname))
    print("Setting up the Schunk SDH meshes is done.")


def setup_wam():
    # Barrett WAM meshes (Pyrado)
    downloadAndExtract(
        wam_url, osp.join(resources_dir, "wam_description"), f"self-paced-rl-{wam_repo_version}/sprl/envs/xml/"
    )

    # Copy the relevant mesh files into Pyrados's MuJoCo environments folder
    src_dir = osp.join(resources_dir, "wam_description/meshes")
    dst_dir = osp.join(pyrado_dir, "pyrado/environments/mujoco/assets/meshes/barrett_wam")

    # Get all .stl files in the wam subdir
    print("Copying the Barrett WAM meshes to the Pyrado assets dir ...")
    for fname in os.listdir(src_dir):
        if fname.endswith(".stl"):
            mkdir_p(osp.dirname(osp.join(dst_dir, fname)))
            shutil.copyfile(osp.join(src_dir, fname), osp.join(dst_dir, fname))
    print("Setting up the Barrett WAM meshes is done.")


def setup_meshes():
    # Set up all external meshes
    setup_iiwa()
    setup_schunk()
    setup_wam()


def setup_pre_commit():
    # Set up pre-commit used for the Black code formatter
    sp.check_call([sys.executable, "-m", "pip", "install", "pre-commit"])
    sp.check_call(["pre-commit", "install"], cwd=osp.join(project_dir, ".github"))


def setup_pyrado():
    # Set up Pyrado in development mode
    sp.check_call([sys.executable, "setup.py", "develop"], cwd=osp.join(project_dir, "Pyrado"))


def setup_mujoco_py():
    # Set up mujoco-py (doing it via pip caused problems on some machines)
    sp.check_call([sys.executable, "setup.py", "install"], cwd=osp.join(project_dir, "thirdParty", "mujoco-py"))


def setup_pytorch_based():
    # Locally build PyTorch==1.7.0 requires dataclasses (does not harm when using pytorch from pip)
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "dataclasses"])
    # Set up GPyTorch without touching the PyTorch installation (requires scikit-learn which requires threadpoolctl)
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "threadpoolctl"])
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "scikit-learn"])
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "gpytorch"])
    # Set up BoTorch without touching the PyTorch installation (requires gpytorch)
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "botorch"])
    # Set up Pyro without touching the PyTorch installation (requires opt-einsum)
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "opt-einsum"])
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "pyro-api"])
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "pyro-ppl"])
    # Set up SBI without touching the PyTorch installation (requires Pyro and pyknos which requires nflows)
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "nflows"])
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "pyknos"])
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "sbi"])
    # Downgrade to avoid the incompatibility with cliff (whatever cliff is)
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "prettytable==0.7.2"])

    if args.pip_check:
        # Check the installations
        print("Checking dependencies of the packages installed via pip:")
        sp.check_call([sys.executable, "-m", "pip", "check"])


def setup_cppsctp():
    # Install dependencies
    required_packages_sctp = [
        "libsctp-dev",
    ]
    user_input = input(
        f"You are about to install SL which depends on the following libraries:"
        f"\n{required_packages_sctp}\nDo you really want this? [y / n] "
    )
    if user_input.lower() == "y":
        sp.check_call(["sudo", "apt-get", "install", "-y"] + required_packages_sctp)
        print("Dependencies have been installed.")
    else:
        print("Dependencies have NOT been installed.")

    # Get it all GitLab
    if not osp.exists(cppsctp_dir):
        mkdir_p(cppsctp_dir)
        sp.check_call(["git", "clone", cppsctp_git_repo, cppsctp_dir])

    # Create relative build dir
    cppsctp_build_dir = osp.join(cppsctp_dir, "build")
    if not osp.exists(cppsctp_build_dir):
        mkdir_p(cppsctp_dir)

    # Build it
    buildCMakeProject(cppsctp_dir, cppsctp_build_dir, cmakeVars=cppsctp_cmake_vars, install_dir=cppsctpinstall_dir)


def setup_sl():
    # Install dependencies (copied from https://git.ias.informatik.tu-darmstadt.de/robcom-2/robcom-2.0/-/wikis/usage)
    required_packages_sl = [
        "libsctp-dev",
        "libncurses5-dev",
        "libreadline6-dev",
        "freeglut3-dev",
        "libxmu-dev",
        "cmake-curses-gui",
        "libedit-dev",
        "clang",
        "xterm",
    ]
    user_input = input(
        f"You are about to install SL which depends on the following libraries:"
        f"\n{required_packages_sl}\nDo you really want this? [y / n] "
    )
    if user_input.lower() == "y":
        sp.check_call(["sudo", "apt-get", "install", "-y"] + required_packages_sl)
        print("Dependencies have been installed.")
    else:
        print("Dependencies have NOT been installed.")

    # Set up custom IAS dependency
    setup_cppsctp()

    # Get it all GitLab
    if not osp.exists(sl_dir):
        mkdir_p(sl_dir)
        sp.check_call(["git", "clone", sl_git_repo, sl_dir])

    # Create relative build dir
    sl_build_dir = osp.join(sl_dir, "build")
    if not osp.exists(sl_build_dir):
        mkdir_p(sl_build_dir)

    # Build it
    buildCMakeProject(sl_dir, sl_build_dir, cmakeVars=sl_cmake_vars)


def setup_robcom():
    # Set up dependency
    setup_cppsctp()

    # Get it all GitLab
    if not osp.exists(robcom_dir):
        mkdir_p(robcom_dir)
        sp.check_call(["git", "clone", robcom_git_repo, robcom_dir])

    # Install it suing its setup script
    env = os.environ.copy()
    env.update(env_vars)
    env["BUILD_ROBCIMPYTHON_WRAPPER"] = "ON"
    env["IAS_DIR"] = ias_dir
    env["INSTALL_IN_IAS_DIR"] = "ON"
    sp.check_call([sys.executable, "setup.py", "install", "--user"], cwd=robcom_dir, env=env)


def setup_wo_rcs_wo_pytorch():
    print("\nStarting Option Red Velvet Setup\n")
    # Rcs will still be downloaded since it is a submodule
    setup_wam()  # ignoring the meshes used in RcsPySim
    setup_mujoco_py()
    if not CI:
        setup_pyrado()
    setup_pytorch_based()
    print("\nWAM meshes, mujoco-py, Pyrado (with GPyTorch, BoTorch, and Pyro using the --no-deps flag) are set up!\n")


def setup_wo_rcs_w_pytorch():
    print("\nStarting Option Malakoff Setup\n")
    # Rcs will still be downloaded since it is a submodule
    setup_pytorch()
    setup_wam()  # ignoring the meshes used in RcsPySim
    setup_mujoco_py()
    if not CI:
        setup_pyrado()
    setup_pytorch_based()
    print(
        "\nPyTorch, WAM meshes, mujoco-py, Pyrado (with GPyTorch, BoTorch, and Pyro using the --no-deps flag) are set up!\n"
    )


def setup_w_rcs_wo_pytorch():
    print("\nStarting Option Sacher Setup\n")
    # We could do setup_dep_libraries() here, but it requires sudo rights
    if not IN_HRI:
        setup_wm5()
    setup_rcs()
    # don't use the local PyTorch but the one from anaconda/pip
    rcspysim_cmake_vars["USE_LIBTORCH"] = "OFF"
    if not CI:
        setup_rcspysim()
    setup_meshes()
    setup_mujoco_py()
    if not CI:
        setup_pyrado()
    setup_pytorch_based()
    print(
        "\nWM5, Rcs, RcsPySim, iiwa & Schunk & WAM meshes, mujoco-py, and Pyrado (with GPyTorch, BoTorch, and Pyro using the --no-deps flag) are set up!\n"
    )


def setup_w_rcs_w_pytorch():
    print("\nStarting Option Black Forest Setup\n")
    # We could do setup_dep_libraries() here, but it requires sudo rights
    if not IN_HRI:
        setup_wm5()
    setup_rcs()
    setup_pytorch()
    if not CI:
        setup_rcspysim()
    setup_meshes()
    setup_mujoco_py()
    if not CI:
        setup_pyrado()
    setup_pytorch_based()
    print(
        "\nWM5, Rcs, PyTorch, RcsPySim, iiwa & Schunk & WAM meshes, mujoco-py, Pyrado (with GPyTorch, BoTorch, and Pyro using the --no-deps flag) are set up!\n"
    )


# All tasks list
tasks_by_name = {name[6:]: v for name, v in globals().items() if name.startswith("setup_")}  # cut the "setup_"

# ==== #
# MAIN #
# ==== #

# Print help if none
if len(args.tasks) == 0:
    print("Available tasks:")
    for n in tasks_by_name.keys():
        print("  {}".format(n))

# Execute selected tasks
for task in args.tasks:
    tasks_by_name[task]()
