#ifndef PYTHON_BINDINGS_H
#define PYTHON_BINDINGS_H

// TODO(brianderson): Should we use pybind directly with the C++ classes,
// rather than the C API?
#include "c_api.h"
#include "../loadgen.h"
#include "third_party/pybind/include/pybind11/pybind11.h"

PYBIND11_MODULE(mlpi_loadgen_lib, m) {
  m.doc() = "MLPerf Inference load generator.";
  m.def("ConstructSUT", &mlperf::c::ConstructSUT,
        "Construct the system under test.");
  m.def("DestroySUT", &mlperf::c::DestroySUT,
        "Destroy the object created by ConstructSUT.");
  m.def("StartTest", &mlperf::c::StartTest,
        "Run tests on a SUT created by ConstructSUT().");
  m.def("QueryComplete", &mlperf::QueryComplete,
        "Called by the SUT to indicate the query_id from the]"
        "AcceptQuery callback is finished.");
}

#endif  // PYTHON_BINDINGS_H
