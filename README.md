# externalSubmissionPOC

## Test cases

**Case-1**: model_maping.json in SUT folder

**Case-2**: model_mapping.json in individual folder

**Case-3**: model_mapping.json not present but model name is matching with the official one in submission checker

**Case-4**: model_mapping.json is not present but model name is mapped to official model name in submission checker. Example: resnet50 to resnet

**Case-5**: Case-1 to Case-4 is not satisfied. The gh action will be successfull if the submission generation fails.

**Case-6**: Case-2 but model_mapping.json is not present in any of the folders. The gh action will be successfull if the submission generation fails.

**Case-7**: sut_info.json is not completely filled but the SUT folder name is in required format(hardware_name-implementation-device-framework-run_config)
