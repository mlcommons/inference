# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Encrypting your project for submission

# In MLPerf Inference v1.0, a policy has been introduced to allow submitters 
# to submit an encrypted tarball of their submission repository along with the 
# decryption password and SHA1 hash of the encrypted tarball to the MLPerf
# Inference results chair. 

# To create an encrypted tarball and generate the SHA1 of the tarball, first 
# change the `SUBMITTER` variable in `scripts/pack_submission.sh` to your 
# company name. Then from the project root, run:

# bash pack_submission.sh --pack

# This command will prompt to enter and then confirm an encryption password. 
# After this command finishes running, there will be 2 files:

# - `mlperf_submission_${SUBMITTER}.tar.gz` - The encrypted tarball, encrypted with AES256
# - `mlperf_submission_${SUBMITTER}.sha1` - A text file containing the sha1 hash of the encrypted tarball

# To test that the submission has been successfully packed, run:

# bash path/to/pack_submission.sh --unpack

# The 3 things that must be shared with the MLPerf Inference results chair for 
# submission are:
# 1. `mlperf_submission_${SUBMITTER}.tar.gz` - The encrypted tarball, encrypted with AES256
# 2. `mlperf_submission_${SUBMITTER}.sha1` - A text file containing the sha1 hash of the encrypted tarball
# 3. The decryption password

# Before submission deadline, upload the tarball to a public cloud storage and 
# email the link along with items 2-3 to the MLCommons submissions address: submissions@mlcommons.org
# Also, include the last two lines of the submission_checker_log.txt like 
# below in the body of the email as cursory evidence of a valid submission. 

# INFO:main:Results=265, NoResults=0
# INFO:main:SUMMARY: submission looks OK


SUBMITTER=COMPANY
TARBALL_NAME=mlperf_submission_${SUBMITTER}.tar.gz
SHA1_FILE_NAME=mlperf_submission_${SUBMITTER}.sha1

if [ "$1" = "--pack" ]; then
    echo "Packing tarball and encrypting"
    tar -cvzf - closed/ open/ | openssl enc -e -aes256 -out ${TARBALL_NAME}
    echo "Generating sha1sum of tarball"
    sha1sum ${TARBALL_NAME} | tee ${SHA1_FILE_NAME}
elif [ "$1" = "--unpack" ]; then
    echo "Checking sha1sum of tarball"
    if [ "`sha1sum ${TARBALL_NAME}`" = "`cat ${SHA1_FILE_NAME}`" ]; then
        echo "sha1sum matches."
        openssl enc -d -aes256 -in ${TARBALL_NAME} | tar -xvz
    else
        echo "ERROR: sha1sum of ${TARBALL_NAME} does not match contents of ${SHA1_FILE_NAME}"
    fi
else
    echo "Unrecognized flag. Must specify --pack or --unpack"
fi

