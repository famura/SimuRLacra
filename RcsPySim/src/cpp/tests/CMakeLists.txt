
set(TESTS_SRCS
    test-main.cpp
    test_env_run.cpp
    test_eigen_matnd.cpp)

IF (USE_LIBTORCH)
  LIST(APPEND TEST_SRCS test_torch.cpp)
ENDIF ()

add_executable(tests ${TESTS_SRCS})
target_link_libraries(tests RcsEnvCore Catch2::Catch2 z)

include(CTest)
include(Catch)
catch_discover_tests(tests)

