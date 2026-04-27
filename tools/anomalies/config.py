SCENARIO_RESULT_KEYS = {
    "Offline": "result_samples_per_second",
    "SingleStream": "early_stopping_latency_ss",
    "MultiStream": "early_stopping_latency_ms",
    "Server": "result_completed_samples_per_sec",
    "Interactive": "result_completed_samples_per_sec",
}

# Overrides SCENARIO_RESULT_KEYS for specific (model, scenario) pairs.
MODEL_SCENARIO_RESULT_KEYS = {
    ("deepseek-r1", "Server"): "result_completed_tokens_per_second",
}

# Maps raw accelerator names to canonical names. Entries are (pattern, canonical) pairs;
# the first case-insensitive substring match wins.
ACCELERATOR_NAME_MAPPINGS = [
    ("AMD Instinct MI355X 288GB HBM3e", "AMD Instinct MI355X 288GB HBM3e"),
    ("AMD Instinct MI355X 288GB HBM3E", "AMD Instinct MI355X 288GB HBM3e"),
    ("AMD Instinct MI350X 288GB HBM3e", "AMD Instinct MI350X 288GB HBM3e"),
    ("AMD Instinct MI350X", "AMD Instinct MI350X 288GB HBM3e"),
    ("AMD Instinct MI325X 256GB HBM3e", "AMD Instinct MI325X 256GB HBM3e"),
    ("AMD Instinct MI325X", "AMD Instinct MI325X 256GB HBM3e"),
    ("AMD Instinct MI300X 192GB HBM3", "AMD Instinct MI300X 192GB HBM3"),
    ("AMD Instinct MI300X", "AMD Instinct MI300X 192GB HBM3"),
    ("MS-Intel Arc Pro B60", "Intel(R) Arc Pro(R) B60"),
    ("Intel(R) Arc Pro(R) B70", "Intel(R) Arc Pro(R) B70"),
    ("Intel(R) Arc Pro(R) B60", "Intel(R) Arc Pro(R) B60"),
    ("Intel(R) Arc Pro(R) B50", "Intel(R) Arc Pro(R) B50"),
    ("NVIDIA B300-SXM", "NVIDIA B300-SXM-270GB"),
    ("NVIDIA B200-SXM", "NVIDIA B200-SXM-180GB"),
    ("NVIDIA GB300", "NVIDIA GB300"),
    ("NVIDIA GB200", "NVIDIA GB200"),
    ("NVIDIA GB10", "NVIDIA GB10"),
    ("NVIDIA B200", "NVIDIA B200-SXM-180GB"),
    ("NVIDIA H200-SXM-141GB", "NVIDIA H200-SXM-141GB"),
    ("NVIDIA H200-NVL-141GB", "NVIDIA H200-NVL-141GB"),
    ("NVIDIA H200", "NVIDIA H200-SXM-141GB"),
    ("NVIDIA L40S", "NVIDIA L40S"),
    ("NVIDIA L4-PCIe-24GB", "NVIDIA L4-PCIe-24GB"),
    ("NVIDIA L4", "NVIDIA L4-PCIe-24GB"),
    ("NVIDIA RTX PRO 6000 Blackwell", "NVIDIA RTX PRO 6000 Blackwell Server Edition"),
    ("NVIDIA RTX PRO 4500 Blackwell", "NVIDIA RTX PRO 4500 Blackwell"),
    ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"),
    ("NVIDIA GeForce GTX 1650 Ti", "NVIDIA GeForce GTX 1650 Ti"),
    ("NVIDIA Jetson AGX Thor 128G", "NVIDIA Jetson AGX Thor 128G"),
]

# Z-score threshold for anomaly detection
ANOMALY_Z_THRESHOLD = 2.5

# Minimum number of historical samples required to run the test
MIN_SAMPLES = 3
