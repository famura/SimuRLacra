
# Module definition
set(MODULE_SRCS
    module.cpp
    config/PropertySourceDict.cpp
    )

# The link-time optimization has an internal error when used from ToolBos, so disable it.
pybind11_add_module(_rcsenv ${MODULE_SRCS} NO_EXTRAS)
target_compile_definitions(_rcsenv PRIVATE "-DRCS_CONFIG_DIR=\"${RCS_CONFIG_DIR}\"")

target_link_libraries(_rcsenv PUBLIC RcsEnvCore)

# Mark pybind11 include dirs as system to supress warnings in pybind11 code
target_include_directories(_rcsenv SYSTEM BEFORE PUBLIC ${PYBIND11_INCLUDE_DIR})

# Setup site-packages link for python build mode
OPTION(SETUP_PYTHON_DEVEL "Add the build output dir to the pythonpath")
IF (SETUP_PYTHON_DEVEL)
  # Find site-packages dir
  execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  SET(RCSPYSIM_PTH "${PYTHON_SITE_PACKAGES_DIR}/rcspysim.pth")
  
  MESSAGE("-- Linking output directory onto python path")
  FILE(WRITE "${RCSPYSIM_PTH}" "\
${LIBRARY_OUTPUT_PATH}
${CMAKE_SOURCE_DIR}/src/python")

ENDIF ()
