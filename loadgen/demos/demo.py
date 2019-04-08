"""
Python demo showing how to use the MLPerf Inference load generator bindings.
"""

from __future__ import print_function
from absl import app
import mlpi_loadgen

def LoadSamplesToRam(query_samples, size_t):
  return

def UnloadSamplesFromRam(query_samples, size_t):
  return

def IssueQuery(query_id, query_samples, sample_count):
  print(sample_count)
  responses = query_samples
  response_count = sample_count
  mlpi_loadgen.QueryComplete(query_id, responses, response_count)

def main(argv):
  sut_name = "DemoSUT"
  sut = mlpi_loadgen.ConstructSUT(sut_name, IssueQuery)
  qsl_name = "DemoQSL"
  qsl = mlpi_loadgen.ConstructQSL(qsl_name, 0, 0, LoadSamplesToRam, UnloadSamplesFromRam)
  command_line = "--mlperf_scenario edge"
  mlpi_loadgen.StartTest(sut, qsl, command_line)
  mlpi_loadgen.DestroyQSL(qsl)
  mlpi_loadgen.DestroySUT(sut)

if __name__ == '__main__':
  app.run(main)
