## Submission Rules

The MLPerf inference submission rules are spread between the [MLCommons policies](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc) and the [MLCommons Inference policies](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) documents. Further, the rules related to power submissions are given [here](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc). 

**The points below are only for ease of use and should never be considered official MLPerf rules.**

## Hardware requirements
1. MLCommons inference results can be submitted on any hardware and we have past results from Raspberry Pi to high-end inference servers.
2. Closed category submission for datacenter category needs **ECC RAM** and also needs to have the **networking** capabilities as detailed [here](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#networking-from-the-v30-round)
3. Power submissions need an [approved power analyzer](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc#74-which-power-analyzers-aka-meters-are-supported).

## Things to Know
 
1. Closed submission needs performance and accuracy run for all the required scenarios (as per edge/datacenter category) with accuracy within 99% or 99.9% as given in the respective task READMEs. Further, the model weights are not supposed to be altered except for quantization. If any of these constraints are not met, the submission cannot go under closed division but can still be submitted under open division.
2. Reference models are mostly fp32 and reference implementations are just for reference and not meant to be directly used by submitters as they are not optimized for performance.
3. Calibration document due **one week** before the submission deadline
4. Power submission needs a power analyzer (approved by SPEC Power) and EULA signature to get access to SPEC PTDaemon

### Is there an automatic way to run the MLPerf inference benchmarks?

MLPerf inference submissions are expected on different hardware and related software stacks. For this reason, only reference implementations are provided by MLCommons and they can guide submitters to make their own optimal implementations for their software/hardware stack. Also, all the previous implementations are made available in the MLCommons Inference results repositories and they can also guide submitters in doing their own implementations.

[The MLCommons taskforce on automation and reproducibility](https://github.com/mlcommons/ck/blob/master/docs/taskforce.md) has automated all the MLCommons inference tasks using the [MLCommons CM language](https://github.com/mlcommons/ck/blob/master/cm) and [this readme](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference) can guide you in running the reference implementations with very minimal effort. Currently, this automation supports MLCommons reference implementations and C++ implementations for Onnxruntime and TFLite. 


### Expected time to do benchmark runs
1. Closed submission under data enter needs offline and server scenario runs with a minimum of ten minutes needed for both. 
2. Closed submission under edge category needs single stream, multi-stream (only for R50 and retinanet), and offline scenarios. A minimum of ten minutes are needed for each scenario. 
3. Further two (three for ResNet50) compliance runs are needed each taking a minimum of 10 minutes for each scenario.
4. SingleStream, MultiStream and Server scenarios use early stopping and so can always finish around 10 minutes
5. Offline scenario needs a minimum of 24756 input queries to be processed -- can take hours for low-performing models like 3dunet, LLMs, etc.
6. Power submission needs an extra ranging mode to determine the peak current usage and this often doubles the overall experiment run time.

## Changes from MLCommons Inference 3.0

1. Two new benchmarks GPT-J and GPT-3 and DLRMv2 replacing DLRM
2. Submission checker is now checking for non-empty README files and mandatory system description and power-related fields
3. New script is provided which can be used to infer scenario results and low-accuracy results from a high-accuracy result
4. `min_query_count` is removed for all scenarios except offline due to early stopping. SingleStream now needs a minimum of 64 queries and MultiStream needs 662 queries as mandated by the early stopping criteria.


