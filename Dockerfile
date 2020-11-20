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

FROM nvidia/cuda:10.1-base-ubuntu18.04


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl ca-certificates sudo git bzip2 libx11-6 \
    gcc g++ make cmake zlib1g-dev swig libsm6 libxext6 \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    wget llvm libncurses5-dev xz-utils tk-dev libxrender1\
    libxml2-dev libxmlsec1-dev libffi-dev libcairo2-dev libjpeg-dev libgif-dev doxygen texlive-base graphviz

RUN adduser --disabled-password --gecos '' --shell /bin/bash user && chown -R user:user /home/user
RUN mkdir /home/user/SimuRLacra && chown user:user /home/user/SimuRLacra
RUN echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/90-pyrado
USER user
WORKDIR /home/user

RUN echo "export PATH=/home/user/miniconda3/bin:$PATH" >> ~/.bashrc
RUN echo "conda activate pyrado" >> ~/.bashrc

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /home/user/miniconda3/bin:$PATH

RUN conda update conda \
    && conda update --all

WORKDIR /home/user/SimuRLacra

RUN conda create -n pyrado python=3.7 blas cmake colorama coverage cython joblib lapack libgcc-ng mkl matplotlib-base numpy optuna pandas patchelf pip pycairo pytest pytest-cov pytest-xdist pyyaml scipy seaborn setuptools sphinx sphinx-math-dollar sphinx_rtd_theme tabulate tqdm -c conda-forge


SHELL ["conda", "run", "-n", "pyrado", "/bin/bash", "-c"]
RUN pip install git+https://github.com/Xfel/init-args-serializer.git@master argparse box2d glfw gym prettyprinter pytest-lazy-fixture tensorboard vpython

ENV PATH /opt/conda/envs/pyrado/bin:$PATH
ENV PYTHONPATH /home/user/SimuRLacra/RcsPySim/build/lib:/home/user/SimuRLacra/Pyrado/:$PYTHONPATH
ENV RCSVIEWER_SIMPLEGRAPHICS 1

RUN git init
COPY Rcs Rcs
COPY thirdParty thirdParty
COPY --chown=user:user RcsPySim RcsPySim
COPY --chown=user:user setup_deps.py .gitmodules ./

RUN ls -la

RUN python setup_deps.py dep_libraries -j8

ARG OPTION=sacher
ARG J=8

COPY --chown=user:user Pyrado Pyrado

RUN if [ $OPTION == 'blackforest' ]; then\
    python setup_deps.py w_rcs_w_pytorch -j$J;\
    fi

RUN if [ $OPTION == 'sacher' ]; then\
    pip install torch==1.7.0\
    && python setup_deps.py w_rcs_wo_pytorch -j$J;\
    fi

RUN if [ $OPTION == 'redvelvet' ]; then\
    pip install torch==1.7.0 &&\
    python setup_deps.py wo_rcs_wo_pytorch -j$J &&\
    rm -fr Rcs RcsPySim;\
    fi

RUN if [ $OPTION == 'malakoff' ]; then\
    python setup_deps.py wo_rcs_w_pytorch -j$J &&\
    rm -fr Rcs RcsPySim;\
    fi

RUN rm -fr .git .gitmodules
