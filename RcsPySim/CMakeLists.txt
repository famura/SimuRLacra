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

cmake_minimum_required(VERSION 3.5)

project(RcsPySim)
SET(BST_INSTALL_CATEGORY Libraries)

# Make sure CMAKE_BUILD_TYPE is set
IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Debug")
ENDIF (NOT CMAKE_BUILD_TYPE)
# Add Rcs's cmake files to module path
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../Rcs/cmake)

# Make sure the libraries land in a subfolder
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Use pybind11 default settings
set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory("${CMAKE_SOURCE_DIR}/../thirdParty/pybind11" "${CMAKE_BINARY_DIR}/pybind11")

# Find Qt
FIND_PACKAGE(Qt5 COMPONENTS Core Gui Widgets QUIET)

# Find RCS dependency
find_package(Rcs REQUIRED
    COMPONENTS RcsCore RcsPhysics
    OPTIONAL_COMPONENTS RcsGraphics RcsGui)

# Optionally depend on pytorch
OPTION(USE_LIBTORCH "Support LibTorch-based control policies")
IF (USE_LIBTORCH)
  # Locate libtorch in python installation
  execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.__file__)"
      OUTPUT_VARIABLE TORCH_INIT_FILE
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  get_filename_component(TORCH_BASEDIR ${TORCH_INIT_FILE} DIRECTORY)
  
  find_package(Torch 1.7 REQUIRED HINTS "${TORCH_BASEDIR}")
  message("-- Building libtorch policy support.")
ENDIF ()

# Use eigen from submodule
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../thirdParty/eigen3")

# Use catch2 for tests
add_subdirectory("${CMAKE_SOURCE_DIR}/../thirdParty/catch2" "${CMAKE_BINARY_DIR}/catch2")
# For some reason, the catch2 contrib cmake files don't get added to the module path if using add_subdirectory. So do it manually.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../thirdParty/catch2/contrib")

# Set some compile flags to improve debuggability (adapted from RCS)
IF (UNIX)
  
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fPIC -rdynamic -Wno-long-long -Wno-variadic-macros -std=c99 -D_XOPEN_SOURCE=600 -ggdb")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -rdynamic -Wno-format -Wno-long-long -Wno-variadic-macros -std=c++14 -ggdb")
  
  IF (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    # setting march to core2 to enable valgrind debugging (also good for Xeon)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=core2")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core2")
    
    # Low level optimization for debug mode, flag for checking stack 
    # corruption, flag for debug output
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -fstack-protector-all")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fstack-protector-all")
  
  ELSE ()
    # setting march to native for optimal performance on local machine
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    
    # Strong optimization for release mode
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
  
  ENDIF ()
  
  # Some special treatment for 32-bit and 64-bit machines
  IF (CMAKE_SIZEOF_VOID_P EQUAL 4)
    
    ADD_DEFINITIONS(-D__32BIT__ -D__linux__)
  
  ELSE ()
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
    ADD_DEFINITIONS(-D__64BIT__ -D__linux__)
  ENDIF ()
  
  # The torch cmake config defines _GLIBCXX_USE_CXX11_ABI, but that messes up the pybind11/numpy integration on GCC 4.8. So we remove it.
  IF (TORCH_FOUND)
    EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    IF (${GCC_VERSION} VERSION_EQUAL "4.8")
      message(STATUS "Removing errorneous definition from torch library")
      set_target_properties(torch PROPERTIES INTERFACE_COMPILE_OPTIONS "")
    ENDIF ()
  ENDIF ()


ENDIF ()

################################################################################
#
# Flags to strip down the file names for logging at compile time. Taken from
# http://stackoverflow.com/questions/8487986/file-macro-shows-full-path
#
################################################################################
IF (NOT WIN32)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"'")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"'")
ENDIF ()

add_subdirectory(src/cpp)

# Doxygen
add_subdirectory(doc)

# If we want to develop Pyrado in CLion, it needs to be in the project hierarchy. This add_subdirectory achives that goal.
add_subdirectory(../Pyrado "${CMAKE_BINARY_DIR}/pyrado")
