cmake_minimum_required(VERSION 3.0)

project(mlperf_loadgen)

# The mlperf_loadgen version.
set(mlperf_loadgen_VERSION_MAJOR 0)
set(mlperf_loadgen_VERSION_MINOR 5)
message("mlperf_loadgen v${mlperf_loadgen_VERSION_MAJOR}.${mlperf_loadgen_VERSION_MINOR}")

# Set build options.
set(CMAKE_CXX_FLAGS "-O2 -W -Wall")
message(STATUS "Using compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Using static linker flags: ${CMAKE_STATIC_LINKER_FLAGS}")
message(STATUS "Using shared linker flags: ${CMAKE_SHARED_LINKER_FLAGS}")

# Output directory for libraries.
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
message(STATUS "Using output path: ${LIBRARY_OUTPUT_PATH}")

# Generate source file with version info.
execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/version_generator.py ${CMAKE_BINARY_DIR}/version_generated.cc ${CMAKE_CURRENT_SOURCE_DIR})

# Add source files.
set(SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/c_api.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/c_api.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/loadgen.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.h
  ${CMAKE_CURRENT_SOURCE_DIR}/test_settings_internal.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/test_settings_internal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/version.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/version.h
  ${CMAKE_BINARY_DIR}/version_generated.cc
)

add_library(mlperf_loadgen STATIC ${SOURCE})
target_link_libraries(mlperf_loadgen)
