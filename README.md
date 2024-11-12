### Clone the Repo
```
git clone -b submission-generation-tests https://github.com/mlcommons/inference.git submission-tests --depth 1
```
### Install cm4mlops
```
pip install cm4mlops
```

### Generate the submission tree
```
cm run script --tags=generate,mlperf,inference,submission \
--results_dir=submission-tests/case-3 \
--run_checker=yes  \
--submission_dir=my_submissions  \
--quiet \
--submitter=MLCommons \
--division=closed
--clean
```
