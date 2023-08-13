# Tools to check Submissions

## `filter_errors.py` (Deprecated)
### Summary
Tool to remove manually verified ERRORs from the log file in the v0.7 submission.

## `generate_final_report.py`
### Inputs
**input**: Path to .csv output file of the [submission checker](#submissioncheckerpy)
### Summary
Generates the spreadsheet with the format in which the final results will be published. This script can be used by running the following command:
```
python3 generate_final_report.py --input <path-to-csv>
```
### Outputs
Spreadsheet with the results.

## `log_parser.py` 
### Summary
Helper module for the submission checker. It parses the logs containing the results of the benchmark.

## `pack_submission.sh` (Deprecated)
### Summary
Creates an encrypted tarball and generate the SHA1 of the tarball. Currently submissions do not need to be encrypted.

## `repository_checks.sh`
### Inputs
Takes as input the path of the directory to run the checks on.
### Summary
Checks that a directory containing one or several submissions is able to be uploaded to github. This script can be used by running the following command:
```
./repository_checks.sh <path-to-folder>
```
### Outputs
Logs in the console the errors that could cause problems uploading the submission to github.

## `submission_checker.py`
### Inputs
**input**: Path to the directory containing one or several submissions.<br>
**version**: Checker version. E.g v1.1, v2.0, v2.1, v3.0, v3.1. <br>
**submitter**: Filter submitters and only run the checks for some specific submitter. <br>
**csv**: Output path where the csv with the results will be stored. E.g `results/summary.csv`. <br>
**skip_compliance**: Flag to skip compliance checks. <br>
**extra-model-benchmark-map**: Extra mapping for model name to benchmarks. E.g `retinanet:ssd-large;efficientnet:ssd-small`<br>
**submission-exceptions**: Flag to ignore errors in submissions<br>

The below input fields are off by default since v3.1 and are mandatory but can be turned on for debugging purposes
**skip-power-check**: Flag to skip the extra power checks. This flag has no effect on non-power submissions <br>
**skip-meaningful-fields-emptiness-check**: Flag to avoid checking if mandatory system description fields are empty
**skip-empty-files-check**: Flag to avoid checking if mandatory measurement files are empty
**skip-check-power-measure-files**: Flag to avoid checking is the requirement power measurement files are present

### Summary
Checks a directory that contains one or several submission. This script can be used by running the following command:
```
python3 submission_checker.py --input <path-to-folder> 
    [--version <version>]
    [--submitter <submitter-name>]
    [--csv <path-to-output>]
    [--skip_compliance]
    [--extra-model-benchmark-map <extra-mapping-string>]
    [--submission-exceptions]
```

### Outputs
- CSV file containing all the valid results in the directory.
- It raises several errors and logs the results that are invalid.

## `truncate_accuracy_log.py`
### Inputs
**input**: Path to directory containing your submission <br>
**output**: Path to the directory to output the submission with truncated files <br>
**submitter**: Organization name <br>
**backup**: Path to the directory to store an unmodified copy of the truncated files <br>
### Summary
Takes a directory containing a submission and truncates `mlperf_log_accuracy.json` files. There are two ways to use this script. First, we could create a new submission directory with the truncated files by running:
```
python truncate_accuracy_log.py --input <original_submission_directory> --submitter <organization_name> --output <new_submission_directory>
```
Second, we could truncate the desired files and place and store a copy of the unmodified files in the backup repository.
```
python tools/submission/truncate_accuracy_log.py --input <original_submission_directory> --submitter <organization_name> --backup <safe_directory> 
```
### Outputs
Output directory with submission with truncated `mlperf_log_accuracy.json` files

## `preprocess_submission.py`
### Inputs
**input**: Path to directory containing your submission <br>
**submitter**: Organization name <br>
### Summary
The input submission directory is modified with empty directories removed and low accuracy results inferred and also multistream and offline scenario results wherever possible. The original input directory is saved in a timestamped directory.
