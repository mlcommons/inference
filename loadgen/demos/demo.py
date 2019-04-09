"""
Python demo showing how to use the MLPerf Inference load generator bindings.
"""

from __future__ import print_function
from absl import app
import mlpi_loadgen

def LoadSamplesToRam(query_samples):
  return

def UnloadSamplesFromRam(query_samples):
  return

def IssueQuery(query_id, query_samples):
  print(query_samples)
  r1 = mlpi_loadgen.QuerySampleResponse(0, 0);
  r2 = mlpi_loadgen.QuerySampleResponse(0, 0);
  r3 = mlpi_loadgen.QuerySampleResponse(0, 0);
  r4 = mlpi_loadgen.QuerySampleResponse(0, 0);
  responses = [r1, r2, r3, r4]
  mlpi_loadgen.QueryComplete(query_id, responses)

def main(argv):
  sut_name = "DemoSUT"
  sut = mlpi_loadgen.ConstructSUT(sut_name, IssueQuery)
  qsl_name = "DemoQSL"
  qsl = mlpi_loadgen.ConstructQSL(qsl_name, 1024, 128, LoadSamplesToRam, UnloadSamplesFromRam)
  command_line = "--mlperf_scenario edge"
  mlpi_loadgen.StartTest(sut, qsl, command_line)
  mlpi_loadgen.DestroyQSL(qsl)
  mlpi_loadgen.DestroySUT(sut)

if __name__ == '__main__':
  app.run(main)
