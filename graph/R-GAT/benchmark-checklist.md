
#### **1. Applicable Categories**
- Datacenter

---

#### **2. Applicable Scenarios for Each Category**
- Offline

---

#### **3. Applicable Compliance Tests**
- TEST01

---

#### **4. Latency Threshold for Server Scenarios**
- Not applicable

---

#### **5. Validation Dataset: Unique Samples**
Number of **unique samples** in the validation dataset and the QSL size specified in 
- [ ] [inference policies benchmark section](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#41-benchmarks)
- [X] [mlperf.conf](https://github.com/mlcommons/inference/blob/master/loadgen/mlperf.conf)
- [X] [Inference benchmark docs](https://github.com/mlcommons/inference/blob/docs/docs/index.md)
  *(Ensure QSL size overflows the system cache if possible.)*

---

#### **6. Equal Issue Mode Applicability**
Documented whether **Equal Issue Mode** is applicable in 
- [X] [mlperf.conf](https://github.com/mlcommons/inference/blob/master/loadgen/mlperf.conf#L42)
- [X] [Inference benchmark docs](https://github.com/mlcommons/inference/blob/docs/docs/index.md)
  *(Relevant if sample processing times are inconsistent across inputs.)*

---

#### **7. Expected Accuracy and `accuracy.txt` Contents**
- [ ] Detailed expected accuracy and the required contents of the `accuracy.txt` file.

---

#### **8. Reference Model Details**
- [ ] Reference model details updated in [Inference benchmark docs](https://github.com/mlcommons/inference/blob/docs/docs/index.md)  

---

#### **9. Reference Implementation Dataset Coverage**
- [ ] Reference implementation successfully processes the entire validation dataset during:
  - [ ] Performance runs
  - [ ] Accuracy runs
  - [ ] Compliance runs  
- [ ] Valid log files passing the submission checker are generated for all runs.

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

#### **12. CPU-Only and Recommended GPU Requirements**
- Can run without GPUs: yes
- Minimum GPU requirements:
  - Number of GPUs: 1
  - Recommended GPU memory: 8GB

---

#### **13. System Memory and Storage Requirements**
- Recommended system requirements:
  - **System RAM** : 512 GB
  - **Storage**: 2.3 TB

---

#### **14. Submission Checker Modifications**
- [X] All necessary changes made to the **submission checker** to validate the benchmark.

---

#### **15. Sample Log Files**
- [ ] Include sample logs for all applicable scenario runs:
  - [ ] `mlperf_log_summary.txt`
  - [ ] `mlperf_log_detail.txt`  
- [ ] Ensure sample logs successfully pass the submission checker and represent compliant runs.
