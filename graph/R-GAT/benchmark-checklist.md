
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
- [] Number of **unique samples** in the validation dataset specified.
- [] QSL size documented.  
  *(Ensure QSL size overflows the system cache.)*

---

#### **6. Equal Issue Mode Applicability**
- [X] Documented whether **Equal Issue Mode** is applicable in the [mlperf.conf](https://github.com/mlcommons/inference/blob/master/loadgen/mlperf.conf#L42) 
  *(Relevant if sample processing times are inconsistent across inputs.)*

---

#### **7. Expected Accuracy and `accuracy.txt` Contents**
- [ ] Detailed expected accuracy and the required contents of the `accuracy.txt` file.

---

#### **8. Reference Model Details**
- [ ] Number of parameters of the model documented.
- [ ] FLOPs (Floating Point Operations) specified.
- [ ] Data type used for determining reference accuracy detailed.  
  *Example: Parameters: 25.6M, FLOPs: 3.8B, Datatype: fp16.*

---

#### **9. Reference Implementation Dataset Coverage**
- [ ] Reference implementation successfully processes the entire validation dataset during:
  - [ ] Performance runs
  - [ ] Accuracy runs
  - [ ] Compliance runs  
- [ ] Valid log files are generated for all runs.

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
- [X] Documented whether the reference implementation can run on **CPUs only**.
- [ ] Minimum GPU requirements specified:
  - [ ] Number of GPUs
  - [ ] Required GPU memory

---

#### **13. System Memory and Storage Requirements**
- [X] Recommended system requirements documented:
  - [X] **System RAM** (e.g., units of 256 GB RAM).
  - [X] **Storage** (e.g., units of 500 GB storage).

---

#### **14. Submission Checker Modifications**
- [X] All necessary changes made to the **submission checker** to validate the benchmark.

---

#### **15. Sample Log Files**
- [ ] Include sample logs for all applicable scenario runs:
  - [ ] `mlperf_log_summary.txt`
  - [ ] `mlperf_log_detail.txt`  
- [ ] Ensure sample logs successfully pass the submission checker and represent compliant runs.
