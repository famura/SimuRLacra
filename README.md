<img alt="logo" align="left" height="150px" src="logo.png" style="padding-right: 20px">

**[Overview](#overview)**  
**[Citing](#citing)**  
**[Installation](#installation)**  
**[Checking](#checking)**   
**[Troubleshooting](#troubleshooting)**


## Overview

SimuRLacra (composed of the two modules Pyrado and RcsPySim) is a Python/C++ framework for reinforcement learning from randomized physics simulations.
The focus is on robotics tasks with mostly continuous control.
It features __randomizable simulations__ written __in standalone Python__ (no license required) as well as simulations driven by the physics engines __Bullet__ (no license required), __Vortex__ (license required), __or MuJoCo__ (license required).


__Pros__  
* __Exceptionally modular treatment of environments via wrappers.__ The key idea behind this was to be able to quickly modify and randomize all available simulation environments. Moreover, SimuRLacra contains unique environments that either run completely in Python or allow you to switch between the Bullet or Vortex (requires license) physics engine.
* __C++ export of policies based on PyTorch Modules.__ You can port your neural-network policies, learned with Python, to you C++ applications. This also holds for stateful recurrent networks and linear policies. 
* __CPU-based parallelization for sampling the environments.__ Similar to the OpenAI Gym, SimuRLacra offers parallelized environments for sampling. This is done by employing [Serializable](https://github.com/Xfel/init-args-serializer), making the simulation environments fully pickleable.
* __Separation of the exploration strategies and the policy.__ Instead of having a GaussianFNN and a GaussianRNN ect. policy, you can wrap your policy architectures with (almost) any exploration scheme. At test time, you simple strip the exploration wrapper.
* __Tested integration of real-world Quanser platforms__. This feature is extremely valuable if you want to conduct sim-to-real research, since you can simply replace the simulated environment with the physical one by changing one line of code.
* __Tested integration of [BoTorch](https://botorch.org/), and [Optuna](https://optuna.org/)__.
* __Detailed documentation__.

__Cons__  
* __No vision-based environments/tasks.__ In principle there is nothing stopping you from integrating computer vision into SimuRLacra. However, I assume there are better suited frameworks out there.
* __Without bells and whistles.__ Most implementations (especially the algorithms) do not focus on performance. After all, this framework was created to understand and prototype things. 
* __Hyper-parameters are not fully tuned.__ Sometimes the most important part of reinforcement learning is the time-consuming search for the right hyper-parameters. I only did this for the environment-algorithm combinations reported in my papers. But, for all the other cases there is [Optuna](https://optuna.org/) and some optuna-based example scripts that you can start from.
* __Unfinished GPU-support.__ At the moment the porting of the policies is implemented but not fully tested. The GPU-enabled re-implementation of the simulation environments in the pysim folder (simple Python simulations) is at question. The environments based on [Rcs](https://github.com/HRI-EU/Rcs) which require the Bullet or Vortex physics engine will only be able to run on CPU.

SimuRLacra was tested on Ubuntu 16.04, 18.04 (recommended), 19.10, and 20.04, with PyTorch 1.3.
The part without C++ dependencies (Pyrado) also works under Windows 10, but is not supported.

__Not the right framework for you?__
* If you are looking for even more modular code or simply want to see how much you can do with Python decorators, check out [vel](https://github.com/MillionIntegrals/vel/tree/master/vel). It is a really beautiful framework.
* If you need code optimized for performance, check out [stable baselines](https://github.com/hill-a/stable-baselines). I know, that was captain obvious.
* If you are missing value-based algorithms will bells and whistles, check out [MushroomRL](https://github.com/MushroomRL/mushroom-rl). The main contributor is good at every sport. Sorry Carlo, but the world has to know it.


## Citing

If you use code or ideas from this project for your research, please cite SimuRLacra.
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

Follow the instructions on the [anaconda homepage](https://www.anaconda.com/download/#download) to download the anaconda (or miniconda) version for your machine (andaconda 3 is recommended).

Clone the repository and go to the project's directory (defaults to SimuRLacra)
```
git clone https://github.com/famura/SimuRLacra.git
# or via ssh
# git clone git@github.com:famura/SimuRLacra.git
cd SimuRLacra
```

Create an anaconda environment (without PyTorch)
```
conda create -n pyrado python=3.7 blas cmake colorama coverage cython joblib lapack libgcc-ng mkl matplotlib-base numpy pandas patchelf pip pycairo pytest pytest-cov pytest-xdist pyyaml scipy seaborn setuptools sphinx sphinx-math-dollar sphinx_rtd_theme tabulate tqdm -c conda-forge

conda activate pyrado

pip install git+https://github.com/Xfel/init-args-serializer.git@master argparse box2d glfw gym pprint pytest-lazy-fixture vpython
```
Any warnings from VPython can be safely ignored.

 _Infrastructure dependent_: install libraries system-wide  
Parts of this framework create Python bindings of [Rcs](https://github.com/HRI-EU/Rcs) called RcsPySim. Running Rcs requires several libraries which can be installed (__requires sudo rights__) via
```
python setup_deps.py dep_libraries
```
This command will install `g++-4.8`, `libqwt-qt5-dev`, `libbullet-dev`, `libfreetype6-dev`, `libxml2-dev`, `libglu1-mesa-dev`, `freeglut3-dev`, `mesa-common-dev`, `libopenscenegraph-dev`, `openscenegraph`, and `liblapack-dev`.
If you can't install the libraries, you can still use the part of this framework which is purely in Python, but no environments in the `rcspysim` folder.
In case you have no sudo rights, but want to use all the Rcs-dependent environments, you can try installing the libraries via anaconda. For references, see the comments behind `required_packages` in `setup_deps.py`.

Now you have __two options__:

1. Create the anaconda environment without PyTorch and build it together with all the other dependencies right after. This is necessary if you want to export PyTorch Modules to C++. The Pyrado `Policy` class is a subclass of PyTorch's `nn.Module`. Finally, the setup script (using the `all`) will install GPyTorch, BoTorch, and Pyro without touching the previously installed PyTroch version.

2. Create the anaconda environment and install PyTorch GPyTorch, BoTorch, and Pyro right after via pip. This version is perfectly fine if you never want to export you learned policies. This variant is faster and we can be sure that PyTorch, BoTorch, ect. are compatible versions.

### Option 1
Next, we will download eigen3, pybind11, catch2, WM5, ect. into the `thirdParty` directory as git submodules. Rcs will be placed in the project's directory. Moreover, we will set up PyTorch.
> _Note_: Following this setup option, PyTorch will be build from source, takes at least 30 min.

Make sure the `pyrado` anaconda environment is activated and run
```
conda activate pyrado
python setup_deps.py all -j8
```
This setup script calls `git submodule init` and `git submodule update`. During the installation of Rcs, the Vortex physics engine as well as the WM5 collision library are searched via a cmake find scripts.
In case this process crashes, please first check the [Troubleshooting](#troubleshooting) section below.

### Option 2
Next, we will download eigen3, pybind11, catch2, WM5, ect. into the `thirdParty` directory as git submodules. Rcs will be placed in the project's directory. Moreover, we will set up PyTorch.
> _Note_: Following this setup option, PyTorch will be installed via anaconda, takes about 2 min.

Make sure the `pyrado` anaconda environment is activated and run
```
conda activate pyrado
python setup_deps.py separate_pytorch -j8
```
This setup script calls `git submodule init` and `git submodule update`. During the installation of Rcs, the Vortex physics engine as well as the WM5 collision library are searched via a cmake find scripts.
In case this process crashes, please first check the [Troubleshooting](#troubleshooting) section below.

### Verify the installation of the anaconda env
```
conda activate pyrado
conda env list
conda list | grep torch  # check if the desired version of PyTorch is installed
python --version  # should return Python 3.7.X :: Anaconda, Inc._
```

### Final notes
If the install script crashes, this is most likely due to missing (or not found, or incomatible) libraries or your OS version. Please see the most common pitfalls below.
In case you are struggeling with the dependencies, or only want to use the Python part, try setting it up using
```
conda activate pyrado
python setup_deps.py no_rcs -j12
```

In the end (given you chose a full install), OpenSceneGraph, eigen3, pybind11, WM5, catch2, Rcs, Kuka iiwa meshes, Schunk SDH meshes, Barrett WAM meshes, PyTorch, mujoco-py, as well as RcsPySim and Pyrado should be downloaded and installed.

>__Optional:__ Create an activation script which also sets the `PYTHONPATH`.
> Replace the `PATH_TO` snippets in the template for activating the anaconda environment with the paths in your system
>```
>gedit PATH_TO/SimuRLacra/Pyrado/activate_pyrado.sh
>```
>and save your custom copy (e.g. in the project's root directory).
>Activate new environment using the custom bash script which 
>```
>source activate_pyrado.sh
>```

### Docker Container (experimental)
There is also a Dockerfile which can be used to spin up a docker container.
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


## Checking

If not already activated, execute
```
conda activate pyrado
cd PATH_TO/SimuRLacra/Pyrado/scripts
```
To exemplarily check basic Pyrado environments (implemented in Python without dependencies to RcsPySim)
```
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
./build_doc.sh
```

RcsPySim
```
firefox RcsPySim/build/doc/html/index.html
```
Pyrado
```
firefox Pyrado/doc/build/index.html
```

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
Set `USE_LIBTORCH = ON` for the cmake arguments of `RcsPySim`
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

If you get visualization errors related to `GLEW` (render causes a frozen window and crashes) add `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so` to your `~/.bashrc` or `~/.zshrc`.
