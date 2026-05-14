# Calibration set selected from Criteo Terabyte dataset

For MLPerf Inference, we use the first 128000 rows (user-item pairs) of the second half of `day_23` as the calibration set. Specifically, `day_23` contains 178274637 rows in total, so we use the rows **from the 89137319-th row to the 89265318-th row (both inclusive) in `day_23`** as the calibration set (assuming 0-based indexing).

Please refer to the [DLRM reference README.md](../../v0.5/recommendation/README.md) for more information.
