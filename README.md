<img alt="logo" align="left" height="175px" src="logo.png" style="padding-right: 20px">

**[Overview](#overview)**  
**[Citing](#citing)**  
**[Installation](#installation)**  
**[Checking](#checking)**   
**[Convenience](#convenience)**   
**[Troubleshooting](#troubleshooting)**


## Overview

SimuRLacra (composed of the two modules Pyrado and RcsPySim) is a Python/C++ framework for reinforcement learning from randomized physics simulations.
The focus is on robotics tasks with mostly continuous control.
It features __randomizable simulations__ written __in standalone Python__ (no license required) as well as simulations driven by the physics engines __Bullet__ (no license required), __Vortex__ (license required), __or MuJoCo__ (license required).

[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation](https://github.com/famura/SimuRLacra/workflows/Documentation/badge.svg?branch=master)](https://famura.github.io/SimuRLacra/)
[![codecov](https://codecov.io/gh/famura/SimuRLacra/branch/master/graph/badge.svg)](https://codecov.io/gh/famura/SimuRLacra)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/imports-isort-green)](https://pycqa.github.io/isort/)

__Pros__  
* __Exceptionally modular treatment of environments via wrappers.__ The key idea behind this was to be able to quickly modify and randomize all available simulation environments. Moreover, SimuRLacra contains unique environments that either run completely in Python or allow you to switch between the Bullet or Vortex (requires license) physics engine.
* __C++ export of policies based on PyTorch Modules.__ Since the `Policy` class is a subclass of PyTorch's `nn.Module`, you can port your neural-network policies, learned with Python, to you C++ applications. This also holds for stateful recurrent networks.
* __CPU-based parallelization for sampling the environments.__ Similar to the OpenAI Gym, SimuRLacra offers parallelized environments for sampling. This is done by employing [Serializable](https://github.com/Xfel/init-args-serializer), making the simulation environments fully pickleable.
* __Separation of the exploration strategies and the policy.__ Instead of having a GaussianFNN and a GaussianRNN ect. policy, you can wrap your policy architectures with (almost) any exploration scheme. At test time, you simple strip the exploration wrapper.
* __Tested integration of real-world Quanser platforms__. This feature is extremely valuable if you want to conduct sim-to-real research, since you can simply replace the simulated environment with the physical one by changing one line of code.
* __Tested integration of [BoTorch](https://botorch.org/), and [Optuna](https://optuna.org/)__.
* __Detailed [documentation](https://famura.github.io/SimuRLacra/)__.

__Cons__  
* __No vision-based environments/tasks.__ In principle there is nothing stopping you from integrating computer vision into SimuRLacra. However, I assume there are better suited frameworks out there.
* __Without bells and whistles.__ The implementations (especially the algorithms) do not focus on performance. After all, this framework was created to understand and prototype things. However, improvement suggestions are always welcome.
* __Hyper-parameters are not fully tuned.__ Sometimes the most important part of reinforcement learning is the time-consuming search for the right hyper-parameters. I only did this for the environment-algorithm combinations reported in my papers. But, for all the other cases there is [Optuna](https://optuna.org/) and some optuna-based example scripts that you can start from.
* __Unfinished GPU-support.__ At the moment the porting of the policies is implemented but not fully tested. The GPU-enabled re-implementation of the simulation environments in the pysim folder (simple Python simulations) is at question. The environments based on [Rcs](https://github.com/HRI-EU/Rcs) which require the Bullet or Vortex physics engine will only be able to run on CPU.

SimuRLacra was tested on Ubuntu 16.04 (deprecated), 18.04 (recommended), and 20.04, with PyTorch 1.4, 1.7 (deprecated) and 1.8.
The part without C++ dependencies, called Pyrado, also works under Windows 10 (not supported).


## Citing

If you use code or ideas from SimuRLacra for your projects or research, please cite it.
```
@misc{Muratore_SimuRLacra,
  author = {Fabio Muratore},
  title = {SimuRLacra - A Framework for Reinforcement Learning from Randomized Simulations},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/famura/SimuRLacra}}
}
```


## Installation

It is recommended to install SimuRLacra in a separate virtual environment such as anaconda. Follow the instructions on the [anaconda homepage](https://www.anaconda.com/download/#download) to download the anaconda (or miniconda) version for your machine (andaconda 3 is recommended).

Clone the repository and go to the project's directory
```
git clone https://github.com/famura/SimuRLacra.git
# or via ssh
# git clone git@github.com:famura/SimuRLacra.git
cd SimuRLacra
```

Create an anaconda environment (without PyTorch) and install the requirements
```
conda create -y -n pyrado python=3.7
conda activate pyrado
conda install -y blas cmake lapack libgcc-ng mkl mkl-include patchelf pip pycairo setuptools -c conda-forge
pip install -r requirements.txt
```


### What do you want to be installed?
If you just want to have a look at SimuRLacra, or don't care about the Rcs-based robotics part, I recommend going for [Red Velvet](#option-red-velvet). However, if you for example want to export your learned controller to a C++ program runnning on a phsical robot, I recommend [Black Forest](#option-black-forest). Here is an overview of the options:

Options                              | PyTorch build | Policy export to C++ | CUDA support       | Rcs-based simulations (RcsPySim)  | Python-based simulations (Pyrado) | (subset of) mujoco-py simulations
---                                  | ---           | ---                  | ---                |  ---                              | ---                               | ---
[Red Velvet](#option-red-velvet)     | pip           | :x:                  | :heavy_check_mark: | :x:                               | :heavy_check_mark:                | :heavy_check_mark: 
[Malakoff](#option-malakoff)         | local         | :heavy_check_mark:   | :x:                | :x:                               | :heavy_check_mark:                | :heavy_check_mark: 
[Sacher](#option-sacher)             | pip           | :x:                  | :heavy_check_mark: | :heavy_check_mark:                | :heavy_check_mark:                | :heavy_check_mark: 
[Black Forest](#option-black-forest) | local         | :heavy_check_mark:   | :x:                | :heavy_check_mark:                | :heavy_check_mark:                | :heavy_check_mark: 

> Please note that the Vortex (optionally used in RcsPySim) as well as the MuJoCo (mandatory for mujoco-py) physics engine require a license.

> Please note that building PyTorch locally from source will take about 30-60 min.

In all cases you will download Rcs, eigen3, pybind11, catch2, and mujoco-py, into the `thirdParty` directory as git submodules. Rcs will be placed in the project's root directory.


### Option Red Velvet
Run (the setup script calls `git submodule init` and `git submodule update`)
```
conda activate pyrado
pip install torch==1.8.1
# or if CUDA support not needed
# pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
python setup_deps.py wo_rcs_wo_pytorch -j8
# or if running headless, e.g., on a computation cluster
# python setup_deps.py wo_rcs_wo_pytorch -j8 --headless
```
In case this process crashes, please first check the [Troubleshooting](#troubleshooting) section below.


### Option Malakoff
Run (the setup script calls `git submodule init` and `git submodule update`)
```
conda activate pyrado
python setup_deps.py wo_rcs_w_pytorch -j8
# or if running headless, e.g., on a computation cluster
# python setup_deps.py wo_rcs_w_pytorch -j8 --headless
```
In case this process crashes, please first check the [Troubleshooting](#troubleshooting) section below.


### Option Sacher
_Infrastructure dependent_: install libraries system-wide  
Parts of this framework create Python bindings of [Rcs](https://github.com/HRI-EU/Rcs) called RcsPySim. Running Rcs requires several libraries which can be installed (__requires sudo rights__) via
```
python setup_deps.py dep_libraries
```
This command will install `g++-4.8`, `libqwt-qt5-dev`, `libbullet-dev`, `libfreetype6-dev`, `libxml2-dev`, `libglu1-mesa-dev`, `freeglut3-dev`, `mesa-common-dev`, `libopenscenegraph-dev`, `openscenegraph`, and `liblapack-dev`.
In case you have no sudo rights, but want to use all the Rcs-dependent environments, you can try installing the libraries via anaconda. For references, see the comments behind `required_packages` in `setup_deps.py`.  
If you can't install the libraries, you can still use the Python part of this framework called Pyrado, but no environments in the `rcspysim` folder.

Run (the setup script calls `git submodule init` and `git submodule update`)
```
conda activate pyrado
pip install torch==1.8.1
# or if CUDA support not needed
# pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
python setup_deps.py w_rcs_wo_pytorch -j8
# or if running headless, e.g., on a computation cluster
# python setup_deps.py w_rcs_wo_pytorch -j8 --headless
```
In case this process crashes, please first check the [Troubleshooting](#troubleshooting) section below.


### Option Black Forest
_Infrastructure dependent_: install libraries system-wide  
Parts of this framework create Python bindings of [Rcs](https://github.com/HRI-EU/Rcs) called RcsPySim. Running Rcs requires several libraries which can be installed (__requires sudo rights__) via
```
python setup_deps.py dep_libraries
```
This command will install `g++-4.8`, `libqwt-qt5-dev`, `libbullet-dev`, `libfreetype6-dev`, `libxml2-dev`, `libglu1-mesa-dev`, `freeglut3-dev`, `mesa-common-dev`, `libopenscenegraph-dev`, `openscenegraph`, and `liblapack-dev`.
In case you have no sudo rights, but want to use all the Rcs-dependent environments, you can try installing the libraries via anaconda. For references, see the comments behind `required_packages` in `setup_deps.py`.  
If you can't install the libraries, you can still use the Python part of this framework called Pyrado, but no environments in the `rcspysim` folder.

Run (the setup script calls `git submodule init` and `git submodule update`)
```
conda activate pyrado
python setup_deps.py w_rcs_w_pytorch -j8
# or if running headless, e.g., on a computation cluster
# python setup_deps.py w_rcs_w_pytorch -j8 --headless
```
In case this process crashes, please first check the [Troubleshooting](#troubleshooting) section below.


### SL & Robcom
In case you are at IAS and want to use you SL and robcom, you can set them up (requires sudo rights) with
```
python setup_deps.py robcom -j8
```
After that you still need to install the robot-specific package in SL.

### Python Code Formatting with Black and isort (and pre-commit)
We are following the Black code style and isort ordering for all Python files. The black package is already integrated to the `pyrado` anaconda environment, and configured by the `pyproject.toml` file.
You can format your local code by running
```
cd PATH_TO/SimuRLacra
isort Pyrado --check --diff  # remove --check to actually do the changes
black Pyrado --check  # remove --check to actually do the changes
```
Moreover, you can install the pre-commit framework via
```
python setup_deps.py pre_commit
```
which will reformat your code before every commit. The conformity with Black and isort is checked using a GitHub action.

<!--
### Docker Container (experimental)
There is also a Dockerfile.blackforest which can be used to spin up a docker container.
Please note that the container is still experimental and not all features have been tested.
Make sure you have Docker installed. If you have not there is a [guide](https://docs.docker.com/engine/install/) on how to install it.
If you want to use cuda inside the container (this does not work on Windows) you need the nvidia-container toolkit which can be installed with one of the following commands depending on the linux distribution.
```
sudo apt-get install -y nvidia-container-toolkit
sudo yum install -y nvidia-container-toolkit
```
Then make sure you have sudo rights and run
```
cd PATH_TO/SimuRLacra
sudo setup_docker.sh
```
Now execute
```
run_docker.sh
```
which opens a shell in the docker with the pyrado virtual env activated.
The command in `run_docker.sh` uses cuda supprort. If you do not want to use cuda remove the `--gpus` option.

It will build the pyrado image. And configure a script to run the docker container with GUI support.
You can also connect the image with IDEs such as PyCharm to develop directly in the docker container.
-->


## Checking

### Verify the installation of PyTorch
```
conda activate pyrado
conda env list
conda list | grep torch  # check if the desired version of PyTorch is installed
python --version  # should return Python 3.7.X :: Anaconda, Inc._
```

### Verify the installation of Pyrado and RcsPySim
To exemplarily check basic Pyrado environments (implemented in Python without dependencies to RcsPySim)
```
conda activate pyrado
cd PATH_TO/SimuRLacra/Pyrado/scripts
python sandbox/sb_qcp.py --env_name qcp-su --dt 0.002
```
Quickly check the environments interfacing Rcs via RcsPySim
```
python sandbox/sb_qq_rcspysim.py
```
If this does not work it may be because Vortex or Bullet is not installed.

For deeper testing, run Pyrado's unit tests
```
cd PATH_TO/SimuRLacra/Pyrado/tests
pytest -v -m "not longtime"
```

### Build and view the documentation
If not already activated, execute
```
conda activate pyrado
```
Build both html documentations
```
cd PATH_TO/SimuRLacra
./build_docs.sh
```
This will fail if you did not set up RcsPySim.

RcsPySim
```
firefox RcsPySim/build/doc/html/index.html
```
Pyrado
```
firefox Pyrado/doc/build/index.html
```


## Convenience

### Handy aliases
You will find yourself often in the same folders, so adding the following aliases to your shell's rc-file will be worth it.
```
alias cds='cd PATH_TO/SimuRLacra'
alias cdps='cd PATH_TO/SimuRLacra/Pyrado/scripts'
alias cdpt='cd PATH_TO/SimuRLacra/Pyrado/data/temp'
alias cdrps='cd PATH_TO/SimuRLacra/RcsPySim/build'
alias cdrcs='cd PATH_TO/SimuRLacra/Rcs/build'
```

### Working on the intersection of C++ and Python (e.g. RcsPySim)
Assuming that you use an IDE (in this case CLion), it is nice to put an empty `CMakeLists.txt` into the Python part of your project (here Pyrado) and include this as a subdirectory from the C++ part of your project by adding
```
add_subdirectory(../Pyrado "${CMAKE_BINARY_DIR}/pyrado")
```
If you then create a project in the RcsPySim directory, your IDE will automatically add Pyrado for you. If you moreover mark Pyrado as `sources root` (CLion specific), it will be parsed by the IDE's git tool.

I also suggest to create run configuration that always build the C++ part (RcsPySim) before executing a Python script.
In CLion or example, you go `Run->Edit Configurations ...`, select `CMake Application`, hit the plus, select `_rcsenv` as target and `python` as executable, make your program arguments a module call like `-m scripts.sandbox.sb_p3l` in connection with the correct working directory `PATH_TO/SimuRLacra/Pyrado`, and most importantly select `Build` in the `Before launch` section.

In a similar fashion, you can directly call Rcs. This is useful when you are creating a new environment and want to iterate the graph xml-file.
In CLion or example, you go `Run->Edit Configurations ...`, select `CMake Application`, hit the plus, select `_rcsenv` as target and `Rcs` as executable, pass Rcs-specific arguments to your program arguments like `-m 4 -dir PATH_TO/SimuRLacra/RcsPySim/config/Planar3Link/ -f gPlanar3Link.xml` in connection with the correct working directory `PATH_TO/SimuRLacra/Rcs/build`, and select `Build` in the `Before launch` section.
There are many more command line arguments for Rcs. Look for `argP` in the Rcs.cpp [source file](https://github.com/HRI-EU/Rcs/blob/master/bin/Rcs.cpp).

### Inspecting training logs
To look at the training report in detail from console, I recommend to put 
```
function pretty_csv {
    column -t -s, -n "$@" | less -F -S -X -K
}
```
into your sell's rc-file. Executing `pretty_csv progress.csv` in the experiments folder will yield a nicely formatted table.
I found this neat little trick on [Stefaan Lippens blog](https://www.stefaanlippens.net/pretty-csv.html). You might need to install `column` depending on your OS.


## Troubleshooting

### Undefined reference to `inflateValidate`
Depending on the libraries install on your machine, you might receive the linker error `undefined reference to inflateValidate@ZLIB_1.2.9` while building Rcs or RcsPySim.
In otder to solve this error, link the z library to the necessary targets by editing the `PATH_TO/SimuRLacra/Rcs/bin/CMakeLists.txt` replacing
```
TARGET_LINK_LIBRARIES(Rcs RcsCore RcsGui RcsGraphics RcsPhysics)
```
by
```
TARGET_LINK_LIBRARIES(Rcs RcsCore RcsGui RcsGraphics RcsPhysics z)
```
and
```
TARGET_LINK_LIBRARIES(TestGeometry RcsCore RcsGui RcsGraphics RcsPhysics)
```
by
```
TARGET_LINK_LIBRARIES(TestGeometry RcsCore RcsGui RcsGraphics RcsPhysics z)
```
The same goes for `PATH_TO/SimuRLacra/Rcs/examples/CMakeLists.txt` where you replace
```
TARGET_LINK_LIBRARIES(ExampleForwardKinematics RcsCore RcsGui RcsGraphics)
```
by
```
TARGET_LINK_LIBRARIES(ExampleForwardKinematics RcsCore RcsGui RcsGraphics z)
```
and 
```
TARGET_LINK_LIBRARIES(ExampleKinetics RcsCore RcsGui RcsGraphics RcsPhysics)
```
by
```
TARGET_LINK_LIBRARIES(ExampleKinetics RcsCore RcsGui RcsGraphics RcsPhysics z)
```

### Python debugger stuck at evaluating expression
By default, the sampling (on CPU) in Pyrado is parallelized using PyTorch's multiprocessing module. Thus, your debuggner will not be connected to the right process. Rerun your script with `num_sampler_envs=1` passed as a parameter to the algorithm, that will then contruct a sampler wich only uses one process.

### Qt5 and Vortex (`libpng15.so`)
If you are using Vortex, which itself has a Qt5-based GUI, RcsPySim may look for the wrong `libpng` version. Make sure that if finds the same one as Rcs (`libpng16.so`) and __not__ the one from Vortex (`libpng15.so`). You can investigate this using the `ldd` (or `lddtree` if installed) command on the generated RcsPySim executables.
An easy fix is to go to your Vortex library directory and move all Qt5-related libs to a newly generated folder, such that they cant be found. This solution is perfectly fine since we are not using the Vortex GUI anyway. Next, clear the `RcsPySim/build` folder and build it again.

### Bullet `double` vs. `float`
Check Rcs with which precision Bullet was build 
```
cd PATH_TO/SimuRLacra/thirdParty/Rcs/build
ccmake .
```
Use the same in RcsPySim
```
cd PATH_TO/SimuRLacra/RcsPySim/build
ccmake . 
```
Rebuild RcsPySim (with activated anaconda env)
```
cd PATH_TO/SimuRLacra/RcsPySim/build
make -j12
```

### Module init-args-initializer
ModuleNotFoundError: No module named 'init_args_serializer'
Install it from
`git+https://github.com/Xfel/init-args-serializer.git@master`

When you export the anaconda environment, the yml-file will contain the line `init-args-serializer==1.0`. This will cause an error when creating a new anaconda environment from this yml-file. To fix this, replace the line with `git+https://github.com/Xfel/init-args-serializer.git@master`.

### PyTorch version
You run a script and get `ImportError: cannot import name 'export'`? Check if your PyTorch version is >= 1.2. If not, update via
```
cd PATH_TO/SimuRLacra
python setup_deps.py pytorch -j12
```
or install the pre-compiled version form anaconda using
```
conda install pytorch torchvision cpuonly -c pytorch
```
__Note:__ if you choose the latter, the C++ export of policies will not work.

### `setup.py` not found
If you receive `PATH_TO/anaconda3/envs/pyrado/bin/python: can't open file 'setup.py': [Errno 2] No such file or directory` while executing `python setup_deps pytorch`, delete the `thirdParty/pytorch` and run
```
cd PATH_TO/SimuRLacra
python setup_deps.py pytorch -j12
```

### Lapack library not found in compile time (PyTorch)
__Option 1:__ if you have sudo rights, run
```
sudo apt-get install libopenblas-dev
```
and then rebuild PyTorch from scratch.
__Option 2:__ if you don't have sudo rights, run
```
conda install -c conda-forge lapack
```
and then rebuild PyTorch from scratch.

### Pyrado's policy export tests are skipped
Run the `setup_deps.py` scripts again with `--local_torch`, or explicitly set `USE_LIBTORCH = ON` for the cmake arguments of `RcsPySim`
```
cd PATH_TO/SimuRLacra/Rcs/build
ccmake .  # set the option, configure (2x), and generate
```

### PyTorch compilation is too slow or uses too many CPUs
The Pytorch setup script (thirdParty/pytorch/setup.py) determines the number of cpus to compile automatically. It can be overridden by setting the environment variable MAX_JOBS:
```
export MAX_JOBS=1
```
Please use your shell syntax accordingly (the above example is for bash).

### Set up MuJoCo and mujoco-py
Download `mujoco200 linux` from the [official page](https://www.roboti.us/index.html) and extract it to `~/.mujoco` such that you have `~/.mujoco/mujoco200`. Put your MuJoCo license file in `~/.mujoco`.

During executing `setup_deps.py`, mujoco-py is set up as a git submodule and installed via the downloaded `setup.py`.
If this fails, have a look at the mujoco-py's [canonical dependencies](https://github.com/openai/mujoco-py/blob/master/Dockerfile). Try again. If you get an error mentioning `patchelf`, run ` conda install -c anaconda patchelf`

In case you get visualization errors related to `GLEW` (render causes a frozen window and crashes, or simply a completely black screen) add `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so` to your shell's rc-file (like `~/.bashrc`).
If you now create a new terminal, it should work. If not, try `sudo apt-get install libglew-dev`.

### Shut down the mujoco-py message about the missing MuJoCo installation / license
If you dont have a MuJoCo license, or MuJoCo is not installed on zour machine, mujoco-py will print an error message. One way to avoid this would be to not install mujoco-py by default. However, this would create even more options above. Thus, we will just fool mujoco-py's checker by creating a fake directory and an empty license file.
```
mkdir /$HOME/.mujoco/mujoco200 -p && touch /$HOME/.mujoco/mjkey.txt
```

### Ubuntu 20.04 and mujoco-py > 2.0.2.5
Uninstall mujoco-py
```
pip uninstall mujoco_py
```
Install mujoco-py version 2.0.2.5 (this time not using the setup.py)
```
pip install mujoco_py==2.0.2.5
```
There might be an error saying `Failed building wheel for mujoco-py` at first, but in the end it should install lockfile and mujoco-py.

### libstdc++.so.6: version `GLIBCXX_3.4.22' not found
This error might come from the scipy.signal.lfilter command (eventually including scipy's fft function). For scipy versions > 1.5.2, this requires GLIBCXX_3.4.22. If your computer is out -of-date and you have no sudo rights, your best option is to set scipy pack to version 1.5.2.
```
conda activate pyrado
conda remove scipy --force
pip install scipy==1.5.2
```

### ImageMagick error from moviepy
Check for the ImageMagick policy file. ImageMagick does not have the proper permission set. You can edit the policy file (requires sudo rights)
```
sudo vi /etc/ImageMagick-6/policy.xml
```
by commenting out the line(s) containing `<policy domain="path" rights="none" pattern="@*" />`. Now try again.
