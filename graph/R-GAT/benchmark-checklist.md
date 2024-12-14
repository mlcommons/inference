
#### **1. Applicable Categories**
- [ ] Edge
- [ ] Datacenter
- [ ] Both

---

#### **2. Applicable Scenarios for Each Category**
- [ ] Single-stream
- [ ] Multi-stream
- [ ] Server
- [ ] Offline

---

#### **3. Applicable Compliance Tests**
- [ ] Compliance tests identified for all applicable categories and scenarios.
- [ ] Confirm **TEST04** is not applicable if processing times vary significantly for different inputs.

---

#### **4. Latency Threshold for Server Scenarios**
- [ ] Documented latency threshold for the **Server** scenario.  
  *(99% of samples must be processed within the specified latency threshold.)*

---

#### **5. Validation Dataset: Unique Samples**
- [ ] Number of **unique samples** in the validation dataset specified.
- [ ] QSL size documented.  
  *(Ensure QSL size overflows the system cache.)*

---

#### **6. Equal Issue Mode Applicability**
- [ ] Documented whether **Equal Issue Mode** is applicable.  
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
- [ ] Verified the reference implementation can perform test runs with a smaller subset of inputs for:
  - [ ] Performance runs
  - [ ] Accuracy runs

---

#### **11. Dataset and Reference Model Instructions**
- [ ] Clear instructions provided for:
  - [ ] Downloading the dataset and reference model.
  - [ ] Using the dataset and model for the benchmark.

---

#### **12. CPU-Only and Recommended GPU Requirements**
- [ ] Documented whether the reference implementation can run on **CPUs only**.
- [ ] Minimum GPU requirements specified:
  - [ ] Number of GPUs
  - [ ] Required GPU memory

---

#### **13. System Memory and Storage Requirements**
- [ ] Recommended system requirements documented:
  - [ ] **System RAM** (e.g., units of 256 GB RAM).
  - [ ] **Storage** (e.g., units of 500 GB storage).

---

#### **14. Submission Checker Modifications**
- [ ] All necessary changes made to the **submission checker** to validate the benchmark.

---

#### **15. Sample Log Files**
- [ ] Include sample logs for all applicable scenario runs:
  - [ ] `mlperf_log_summary.txt`
  - [ ] `mlperf_log_detail.txt`  
- [ ] Ensure sample logs successfully pass the submission checker and represent compliant runs.
