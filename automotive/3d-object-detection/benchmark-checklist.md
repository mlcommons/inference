
#### **1. Applicable Categories**
- Edge

---

#### **2. Applicable Scenarios for Each Category**
- SingleStream

---

#### **3. Applicable Compliance Tests**
- TEST01
- TEST04

---

#### **4. Latency Threshold for Server Scenarios**
- Not applicable

---

#### **5. Validation Dataset: Unique Samples**
Number of **unique samples** in the validation dataset and the QSL size specified in 
- [X] [inference policies benchmark section](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#41-benchmarks)
- [X] [mlperf.conf](https://github.com/mlcommons/inference/blob/master/loadgen/mlperf.conf)
- [X] [Inference benchmark docs](https://github.com/mlcommons/inference/blob/docs/docs/index.md)
  *(Ensure QSL size overflows the system cache if possible.)*

---

#### **6. Equal Issue Mode Applicability**
Documented whether **Equal Issue Mode** is applicable in 
- [X] [mlperf.conf](https://github.com/mlcommons/inference/blob/master/loadgen/mlperf.conf)
- [X] [Inference benchmark docs](https://github.com/mlcommons/inference/blob/docs/docs/index.md)
  *(Relevant if sample processing times are inconsistent across inputs.)*

---

#### **7. Expected Accuracy and `accuracy.txt` Contents**
- [X] Detailed expected accuracy and the required contents of the `accuracy.txt` file [here](https://github.com/mlcommons/mlperf_inference_unofficial_submissions_v5.0/blob/auto-update/open/MLCommons/results/mlc-server-reference-gpu-pytorch_v2.2.2-cu124/pointpainting/singlestream/accuracy/accuracy.txt).

---

#### **8. Reference Model Details**
- [ ] Reference model details updated in [Inference benchmark docs](https://github.com/mlcommons/inference/blob/docs/docs/index.md)  

---

#### **9. Reference Implementation Test Coverage**
- [X] Reference implementation successfully does:
  - [X] Performance runs
  - [X] Accuracy runs
  - [X] Compliance runs  


---

#### **10. Test Runs with Smaller Input Sets**
- [X] Verified the reference implementation can perform test runs with a smaller subset of inputs for:
  - [X] Performance runs
  - [X] Accuracy runs

---

#### **11. Dataset and Reference Model Instructions**
- [X] Clear instructions provided for:
  - [X] Downloading the dataset and reference model.
  - [X] Using the dataset and model for the benchmark.

---

#### **12. Documentation of Recommended System Requirements to run the reference implementation**
- [ ] Added [here](https://github.com/mlcommons/inference/blob/docs/docs/system_requirements.yml)

---

#### **13. Submission Checker Modifications**
- [ ] All necessary changes made to the **submission checker** to validate the benchmark.

---

#### **14. Sample Log Files**
- [ ] Include sample logs for all the applicable scenario runs:
  - [ ] `mlperf_log_summary.txt`
  - [ ] `mlperf_log_detail.txt`  
- [X] Log files passing the submission checker are generated for all Divisions.
  - [X] Closed
  - [X] Open  
