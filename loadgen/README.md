# Overview {#mainpage}

* The LoadGen is a *reusable* module that *efficiently* and *fairly* measures
  the performance of inference systems.
* It generates traffic for scenarios as formulated by a diverse set of experts
  in the [MLPerf working group](https://mlperf.org/about).
* The scenarios emulate the types of workloads seen in mobile devices,
  autonomous vehicles, robotics, and cloud-based setups.
* Although the LoadGen is not model or dataset aware, its strength is in its
  reusability with logic that is.

## Useful Links
* [FAQ](@ref ReadmeFAQ)
* [LoadGen Build Instructions](@ref ReadmeBuild)
* [LoadGen API](@ref LoadgenAPI)
* [Test Settings](@ref LoadgenAPITestSettings) -
  A good description of available scenarios, modes, and knobs.
* [MLPerf Inference Code](https://github.com/mlperf/inference) -
  Includes the LoadGen and reference models that use the LoadGen.
* [MLPerf Inference Rules](https://github.com/mlperf/inference_policies) -
  Any mismatch with this is a bug in the LoadGen.
* [MLPerf Website](www.mlperf.org)

## Scope of the LoadGen's Responsibilities

### In Scope
* **Provide a reusable** C++ library with python bindings.
* **Implement** the traffic patterns of the MLPerf inference scenarios and
  modes.
* **Record** all traffic generated and received for later analysis and
  verification.
* **Summarize** the results and whether the scenario's performance constraints
  were met.
* **Target high-performance** systems with an efficient multi-thread friendly
  logging infrastructure.
* **Generate trust** via a well-tested and community-hardened code base.

### Out of Scope
* **It is not** aware of the ML model it is running against.
* **It is not** aware of the data formats of the model's inputs and outputs.
* **It is not** aware of how to score the acuracy of a model's outputs.
* **It is not** aware of MLPerf rules regarding scenario-specific constraints.

Limitting the scope of the LoadGen in this way keeps it reusable across
different models and datasets without modification. Using composition and
dependency injection, the user can define their own model, datasets, and
metrics.

Additionally, not hardcoding MLPerf-specific test constraints, like test
duration and performance targets, allows users to use the LoadGen unmodified
for custom testing and continuous integration purposes.

## Submission Considerations

### Upstream all local modifications
* As a rule, no local modifications to the LoadGen's C++ library are allowed
for submission.
* Please upstream early and often to keep the playing field level.

### Choose your TestSettings carefully!
* Since the LoadGen is oblivious to the model, it can't enforce the MLPerf
requirements for submission. *e.g.:* target percentiles and latencies.
* For verification, the values in TestSettings are logged.
* *Note:* There is an effort to load TestSettings from a config file to make
it easier to respect submission requirements, but it is a work in progress.

## Responsibilities of a LoadGen User

There are numerous eamples in demos, tests, and reference models of code
that integrates with the LoadGen that provide detailed templates for you to
follow. At a high-level, to use the LoadGen, a user must:

* Implement the Iterfaces
  + Implement the SystemUnderTest and QuerySampleLibrary interfaces and pass
    them to the StartTest function.
  + Call QuerySampleComplete for every sample received in IssueQuery.

* Assess Accuracy
  + Process the mlperf_accuracy_log.json output by the LoadGen to determine
    the accuracy of your system.
  + Python scripts will be provided by MLPerf model owners for you to do
    this automatically for the official models.
