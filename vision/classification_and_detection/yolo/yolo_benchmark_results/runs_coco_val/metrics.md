## Benchmark Results: yolo11n/s/m

* **Host:** `macOS-26.0.1-arm64-arm-64bit` (Processor: `arm`)
* **Dataset:** COCO 2017 Validation (5000 images)
* **Models:** `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`

| Model | Format | mAP50-95 | mAP50 | Inf. Speed (ms) | Throughput (img/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `yolo11n` | CoreML | 0.4658 | 0.5982 | 11.44 | 54.97 |
| `yolo11n` | ONNX | 0.4662 | 0.5993 | 37.86 | 22.34 |
| `yolo11s` | CoreML | 0.5252 | 0.6634 | 13.75 | 46.17 |
| `yolo11s` | ONNX | 0.5262 | 0.6638 | 86.39 | 10.47 |
| `yolo11m` | CoreML | 0.5639 | 0.6983 | 22.17 | 33.50 |
| `yolo11m` | ONNX | 0.5650 | 0.6984 | 222.41 | 4.28 |