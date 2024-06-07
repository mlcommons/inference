If you follow the `cm run` commands under the individual model pages in the [benchmarks](../benchmarks/index.md) directory, all the valid results will get aggregated to the `cm cache` folder. Once all the results across all the modelsare ready you can use the following command to generate a valid submission tree compliant with the [MLPerf requirements](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc#inference-1).

## Generate actual submission tree

=== "Closed Edge"
    ### Closed Edge Submission
    ```bash
       cm run script -tags=generate,inference,submission \
      --clean \
      --preprocess_submission=yes \
      --run-checker \
      --submitter=MLCommons \
      --tar=yes \
      --env.CM_TAR_OUTFILE=submission.tar.gz \
      --division=closed \
      --category=edge \
      --env.CM_DETERMINE_MEMORY_CONFIGURATION=yes \
      --quiet
    ```

=== "Closed Datacenter"
    ### Closed Datacenter Submission
    ```bash
       cm run script -tags=generate,inference,submission \
      --clean \
      --preprocess_submission=yes \
      --run-checker \
      --submitter=MLCommons \
      --tar=yes \
      --env.CM_TAR_OUTFILE=submission.tar.gz \
      --division=closed \
      --category=datacenter \
      --env.CM_DETERMINE_MEMORY_CONFIGURATION=yes \
      --quiet
    ```
=== "Open Edge"
    ### Open Edge Submission
    ```bash
       cm run script -tags=generate,inference,submission \
      --clean \
      --preprocess_submission=yes \
      --run-checker \
      --submitter=MLCommons \
      --tar=yes \
      --env.CM_TAR_OUTFILE=submission.tar.gz \
      --division=open \
      --category=edge \
      --env.CM_DETERMINE_MEMORY_CONFIGURATION=yes \
      --quiet
    ```
=== "Open Datacenter"
    ### Closed Datacenter Submission
    ```bash
       cm run script -tags=generate,inference,submission \
      --clean \
      --preprocess_submission=yes \
      --run-checker \
      --submitter=MLCommons \
      --tar=yes \
      --env.CM_TAR_OUTFILE=submission.tar.gz \
      --division=open \
      --category=datacenter \
      --env.CM_DETERMINE_MEMORY_CONFIGURATION=yes \
      --quiet
    ```

* Use `--hw_name="My system name"` to give a meaningful system name. Examples can be seen [here](https://github.com/mlcommons/inference_results_v3.0/tree/main/open/cTuning/systems)

* Use `--submitter=<Your name>` if your organization is an official MLCommons member and would like to submit under your organization

* Use `--hw_notes_extra` option to add additional notes like `--hw_notes_extra="Result taken by NAME" `

The above command should generate "submission.tar.gz" if there are no submission checker issues and you can upload it to the [MLCommons Submission UI](https://submissions-ui.mlcommons.org/submission).

## Aggregate Results in GitHub

If you are collecting results across multiple systems you can generate different submissions and aggregate all of them to a GitHub repository (can be private) and use it to generate a single tar ball which can be uploaded to the [MLCommons Submission UI](https://submissions-ui.mlcommons.org/submission). 

Run the following command after **replacing `--repo_url` with your GitHub repository URL**.

```bash
   cm run script --tags=push,github,mlperf,inference,submission \
   --repo_url=https://github.com/GATEOverflow/mlperf_inference_submissions_v4.1 \
   --commit_message="Results on <HW name> added by <Name>" \
   --quiet
```

At the end, you can download the github repo and upload to the [MLCommons Submission UI](https://submissions-ui.mlcommons.org/submission).
