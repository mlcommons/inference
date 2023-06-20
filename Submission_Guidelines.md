## Submission Rules

The MLPerf inference submission rules are spread between the [MLCommons policies](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc) and the [MLCommons Inference policies](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) documents. Further, the rules related to power submissions are given [here](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc). The below points are a summary taken from the official rules to act as a checklist for the submitters.


## Hardware requirements
1. MLCommons inference results can be submitted on any hardware and we have past results from Raspberry Pi to high-end inference servers.
2. Closed category submission for datacenter category needs **ECC RAM** and also needs to have the **networking** capabilities as detailed [here](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#networking-from-the-v30-round)
3. Power submissions need an [approved power analyzer](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc#74-which-power-analyzers-aka-meters-are-supported).

## Things to Know
 
1. Closed submission needs performance and accuracy run for all the required scenarios (as per edge/datacenter category) with accuracy within 99% or 99.9% as given in the respective task READMEs. Further, the model weights are not supposed to be altered except for quantization. If any of these constraints are not met, the submission cannot go under closed division but can still be submitted under open division.
2. Reference models are mostly fp32 and reference implementations are just for reference and not meant to be directly used by submitters as they are not optimized for performance.
3. Calibration document due **one week** before the submission deadline
4. Power submission needs a power analyzer (approved by SPEC Power) and EULA signature to get access to SPEC PTDaemon
5. To submit under the `available` category your submission system must be available (in whole or in parts and either publicly or to customers) and the software used must be either open source or an **official or beta release** as on the submission deadline. Submissions using nightly release for example cannot be submitted under the available category. 

### Is there an automatic way to run the MLPerf inference benchmarks?

MLPerf inference submissions are expected on different hardware and related software stacks. For this reason, only reference implementations are provided by MLCommons and they can guide submitters to make their own optimal implementations for their software/hardware stack. Also, all the previous implementations are made available in the MLCommons Inference results repositories and they can also guide submitters in doing their own implementations.

[The MLCommons taskforce on automation and reproducibility](https://github.com/mlcommons/ck/blob/master/docs/taskforce.md) has automated all the MLCommons inference tasks using the [MLCommons CM language](https://github.com/mlcommons/ck/blob/master/cm) and [this readme](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference) can guide you in running the reference implementations with very minimal effort. Currently, this automation supports MLCommons reference implementations, Nvidia implementations, and C++ implementations for Onnxruntime and TFLite. Feel free to join the [taskforce Discord channel](https://discord.gg/8jbEM4J6Ff) if you have any questions.

The previous MLPerf inference results are aggregated in [Collective Knowledge platform (MLCommons CK playground)](platform) as [reproducible experiments](https://access.cknowledge.org/playground/?action=experiments)  and can be used by submitters to compare their results with the previous ones while adding various derived metrics (such as performance/watt) and constraints.

### Expected time to do benchmark runs
1. Closed submission under data enter needs offline and server scenario runs with a minimum of ten minutes needed for both. 
2. Closed submission under edge category needs single stream, multi-stream (only for R50 and retinanet), and offline scenarios. A minimum of ten minutes are needed for each scenario. 
3. Further two (three for ResNet50) compliance runs are needed for closed division, each taking at least 10 minutes for each scenario.
4. SingleStream, MultiStream and Server scenarios use early stopping and so can always finish around 10 minutes
5. Offline scenario needs a minimum of 24756 input queries to be processed -- can take hours for low-performing models like 3dunet, LLMs, etc.
6. Open division has no accuracy constraints, no compliance runs required, and can be submitted for any single scenario as well. There is no constraint on the model used also except that the model must be trained on the dataset used in the corresponding MLPerf inference task.
7. Power submission needs an extra ranging mode to determine the peak current usage and this often doubles the overall experiment run time.


## Validity of the submission

1. [MLCommons Inference submission checker](https://github.com/mlcommons/inference/blob/master/tools/submission/submission_checker.py) is provided to ensure that all submissions are passing the required checks.
2. In the unlikely event that there is an error on the submission checker for your submission, please raise a Github issue [here](https://github.com/mlcommons/inference/issues)
3. Any submission passing the submission checker is valid to go to the review discussions but submitters are still required to answer any queries and fix any issues being reported by other submitters.

### Reviewing other submissions
1. Ensure that the `system_desc_id.json` file is having meaningful responses - submission_checker only checks for the existence of the fields.
2. For power submissions, `power settings` and `analyzer table` files are to be submitted, and even though the submission checker checks for the existence of these files, the content of [these files](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc#64-power-management-settings) must be checked manually for validity.
3. README files in the submission directory must be checked to make sure that the instructions are reproducible.
4. For closed datacenter submissions, [ECC RAM and Networking requirements](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#constraints-for-the-closed-division) must be ensured.
5. Submission checker might be reporting warnings and some of these warnings can warrant an answer from the submitter.

## Changes from MLCommons Inference 3.0

1. Two new benchmarks GPT-J and GPT-3 and DLRMv2 replacing DLRM
2. Submission checker is now checking for non-empty README files and mandatory system description and power-related fields
3. New script is provided which can be used to infer scenario results and low-accuracy results from a high-accuracy result
4. `min_query_count` is removed for all scenarios except offline due to early stopping. SingleStream now needs a minimum of 64 queries and MultiStream needs 662 queries as mandated by the early stopping criteria.


