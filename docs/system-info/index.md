# MLPerf Inference System Info Collection

This guide covers how to automatically collect hardware and software information from one or more nodes for MLPerf Inference submissions using the MLC `get-mlperf-multi-node-system-info` script.

!!! note "MLPerf Inference v6.1 scope"
    For inference round v6.1, the sysinfo tool aims to automate system inventory collection (CPU, GPU, memory, storage, OS and software stack). Power and Network mode fields are scoped for future rounds and scaffolded as empty strings in the output. Submitters have to fill them in manually before submission.

## Prerequisites

**Install MLC** — Follow the [MLC installation guide](../install/index.md) to set up `mlc-scripts`.

**SSH access** — For multi-node setups the script SSHes into each remote node to run the hardware probe. Password-based SSH works, but passwordless (key-based) access is strongly recommended — it is seamless and avoids repeated prompts when probing a large number of nodes.

To set up passwordless SSH:

```bash
# Generate a key if you don't have one
ssh-keygen -t rsa -b 4096

# Copy it to every target node
ssh-copy-id user@node1
ssh-copy-id user@node2

# Verify
ssh user@node1 "hostname && nvidia-smi -L"
```

## Quick Start

The same `get-mlperf-multi-node-system-info` script handles everything from a single node to a large cluster. Internally it always calls `get-mlperf-single-node-system-info` on each target node (including the local machine as node 0). When no `--ssh_ids` are given it probes only the local machine.

**Single node (local machine)**

```bash
mlcr get-mlperf-multi-node-system-info,_cuda,_inference \
  --system_name="My-1xH100-System"
```

**Multi-node cluster**

```bash
mlcr get-mlperf-multi-node-system-info,_cuda,_inference \
  --ssh_ids="user@node1:22,user@node2:22,user@node3:22" \
  --system_name="24xH100-Cluster"
```

The local machine is included as node 0; each SSH target becomes node 1, 2, 3, and so on. To collect from SSH targets only and exclude the local machine:

```bash
mlcr get-mlperf-multi-node-system-info,_cuda,_inference,_exclude_current_node \
  --ssh_ids="user@node1:22,user@node2:22" \
  --system_name="Remote-Only-Cluster"
```

## Using a Config File

For repeated runs or shared team configs, store submission metadata in a YAML or JSON file instead of passing many CLI flags. CLI arguments always take precedence over config file values.

```yaml
# system_config.yaml
submitter_org_names: "Your Organization"
system_name: "8xH100-vLLM-Server"
division: "open"
system_category: "datacenter"
system_availability_status: "available"
cooling: "air"
hw_notes: "DGX H100 node, 8x NVLink-connected H100 SXM5"
system_type_detail: "on-premise"
```

```bash
mlcr get-mlperf-multi-node-system-info,_cuda,_inference \
  --config_file=system_config.yaml \
  --ssh_ids="user@node1:22,user@node2:22"
```



## Serving Framework Detection

`framework` is a required field for the MLPerf Inference submission. The script can detect it automatically in two ways.

**Auto-detect via HTTP probe** — provide `--endpoint_url` and the script probes the running inference server:

```bash
mlcr get-mlperf-multi-node-system-info,_cuda,_inference \
  --ssh_ids="user@node1:22" \
  --endpoint_url="http://node1:8000" \
  --system_name="vLLM-System"
```

| Framework | Endpoint probed |
|-----------|----------------|
| TRT-LLM | `/perf_metrics` |
| vLLM | `/version` |
| SGLang | `/get_server_info` |


## Output

The script writes `system-info-multi-node.json` in the current directory. The full path is printed at the end of the run and exported as `MLC_MULTI_NODE_SYSTEM_INFO_FILE_PATH`.

With `_inference`, the output is a flat JSON as required for the MLPerf Inference submission. Submitters must verify the generated `system-info-multi-node.json` and manually fill in any fields that are empty. Fields expected to be auto-detected (see [Hardware and Software Fields](#hardware-and-software-fields) below) should not be empty — if any of those come out as an empty string, please [raise an issue](https://github.com/mlcommons/mlperf-automations/issues) with the field name and details of your machine.

When the `_network` variation is also active, all network mode fields are appended to the output as empty strings and must be filled in manually — see [Network Mode Fields](#network-mode-fields-with-_network-variation) for the full list.

```json
{
  "submitter": "Your Organization",
  "system_name": "2xDGX-H100-vLLM",
  "status": "available",
  "system_type": "datacenter",
  "division": "open",
  "system_size": "16x NVIDIA H100 80GB HBM3",
  "number_of_nodes": 2,
  "host_processor_model_name": "Intel(R) Xeon(R) Platinum 8480+",
  "host_processors_per_node": 2,
  "host_processor_core_count": 112,
  "host_processor_vcpu_count": 224,
  "host_processor_frequency": "3.80 GHz",
  "host_processor_caches": "L1d: 4.4 MiB; L1i: 2.2 MiB; L2: 224 MiB; L3: 210 MiB",
  "host_processor_interconnect": "",
  "host_memory_capacity": "2.2T",
  "host_storage_type": "NVMe SSD",
  "host_storage_capacity": "1.8 TB SSD",
  "host_memory_configuration": "DDR5",
  "host_networking": "mlx5_0: native InfiniBand",
  "host_networking_topology": "",
  "host_network_card_count": "3x mlx5_0: native InfiniBand",
  "accelerator_model_name": "NVIDIA H100 80GB HBM3",
  "accelerators_per_node": 8,
  "accelerator_memory_capacity": "80GiB",
  "accelerator_memory_configuration": "80 GiB HBM3",
  "accelerator_host_interconnect": "PCIe Gen5 x16",
  "accelerator_interconnect": "NVLink",
  "accelerator_interconnect_topology": "",
  "accelerator_frequency": "",
  "accelerator_on-chip_memories": "Shared Memory: 228 KB/block",
  "framework": "vLLM 0.4.3",
  "operating_system": "ubuntu 24.04",
  "other_software_stack": "CUDA 12.9, Driver 575.57.08",
  "hw_notes": "",
  "sw_notes": "",
  "other_hardware": "",
  "cooling": "air",
  "system_type_detail": ""
}
```

## Information Captured

### Hardware and Software Fields

These are collected automatically on each node. If a value cannot be detected (driver missing, command unavailable), the field is set to `""` or `"N/A"` in the output for manual completion.

| Field | Description | Auto-Detected |
|-------|-------------|:---:|
| `host_processor_model_name` | CPU model name | ✅ |
| `host_processors_per_node` | Number of CPU sockets | ✅ |
| `host_processor_core_count` | Physical CPU cores per socket | ✅ |
| `host_processor_frequency` | CPU maximum frequency | ✅ |
| `host_processor_caches` | L1d / L1i / L2 / L3 cache sizes | ✅ |
| `host_processor_interconnect` | CPU-to-CPU interconnect inferred from NUMA topology | ✅ |
| `host_memory_capacity` | Total system RAM | ✅ |
| `host_memory_configuration` | Memory type and speed | ✅ |
| `host_storage_type` | Primary storage type (NVMe, SSD, HDD) | ✅ |
| `host_storage_capacity` | Total disk capacity | ✅ |
| `host_networking` | Primary NIC description | ✅ |
| `host_network_card_count` | NIC count and model | ✅ |
| `accelerator_model_name` | GPU model name | ✅ |
| `accelerators_per_node` | Number of GPUs per node | ✅ |
| `accelerator_memory_capacity` | GPU memory per device | ✅ |
| `accelerator_memory_configuration` | GPU memory size and type | ✅ |
| `accelerator_host_interconnect` | Host-to-GPU link (PCIe Gen, NVLink) | ✅ |
| `accelerator_interconnect` | GPU-to-GPU link (NVLink, xGMI) | ✅ |
| `accelerator_interconnect_topology` | Interconnect topology description | ⚠️ CUDA only; may be empty |
| `accelerator_frequency` | GPU clock frequency | ✅ |
| `accelerator_on-chip_memories` | Shared memory per SM block | ✅ |
| `operating_system` | OS distribution and version | ✅ |
| `other_software_stack` | CUDA/ROCm version + driver version | ✅ |
| `number_of_nodes` | Total node count | ✅ Computed from SSH targets |
| `framework` | Inference framework and version | ⚠️ Requires `--endpoint_url` |

### Submission Identity Fields

These fields are not detectable from hardware and must be supplied via CLI flags, a config file, or by directly editing the output JSON before submission.

| Field | CLI Flag | Config Key | Notes |
|-------|----------|-----------|-------|
| `system_name` | `--system_name` | `system_name` | Required |
| `submitter` | `--submitter_org_names` | `submitter_org_names` | Required |
| `division` | `--division` | `division` | `open` or `closed` |
| `status` | `--system_availability_status` | `system_availability_status` | System availability — `available` (publicly available), `preview` (available soon), or `rdi` (Research, Development, and Internal use only) |
| `system_type` | `--category` | `system_category` | `datacenter` or `edge` |
| `cooling` | `--cooling` | `cooling` | e.g. `air`, `liquid` |
| `hw_notes` | `--hw_notes` | `hw_notes` | Hardware notes |
| `sw_notes` | *(manual edit)* | — | Software notes; fill in the output JSON |
| `host_networking_topology` | *(manual edit)* | — | Network topology description (not auto-detected; different from `host_networking` which is captured automatically) |
| `system_type_detail` | `--system_type_detail` | `system_type_detail` | More specific system type — `cloud`, `on-premise`, `edge-server`, or `edge-device` (optional) |

### Network Mode Fields (with `_network` variation)

When the `_network` variation is active alongside `_inference`, the script adds the fields required for network mode submissions as per MLPerf Inference submission rules. All fields are initialised to `""` and must be filled in manually.

`is_network`, `network_type`, `network_media`, `network_rate`, `nic_loadgen`, `number_nic_loadgen`, `net_software_stack_loadgen`, `network_protocol`, `number_connections`, `nic_sut`, `number_nic_sut`, `net_software_stack_sut`, `network_topology`

### Power Measurement Fields (with `_power` variation)

When the `_power` variation is active, the following additional fields are required. All are initialised to `""` and must be filled in manually.

`power_management`, `filesystem`, `boot_firmware_version`, `management_firmware_version`, `other_hardware`, `number_of_type_nics_installed`, `nics_enabled_firmware`, `nics_enabled_os`, `nics_enabled_connected`, `network_speed_mbit`, `power_supply_quantity_and_rating_watts`, `power_supply_details`, `disk_drives`, `disk_controllers`, `system_power_only`

## Available Variations

| Variation | Description |
|-----------|-------------|
| `_cuda` | Probe NVIDIA GPUs via CUDA |
| `_rocm` | Probe AMD GPUs via ROCm |
| `_xpu` | Probe Intel GPUs via XPU |
| `_inference` | Flat JSON output as required for MLPerf Inference submission |
| `_exclude_current_node` | Skip the local machine; collect only from SSH targets |
| `_network` | Add network mode fields to the output |
| `_power` | Add power measurement fields to the output |

Variations can be stacked:

```bash
mlcr get-mlperf-multi-node-system-info,_cuda,_inference,_power \
  --ssh_ids="user@node1:22" \
  --system_name="My-System"
```

## Key Parameters Reference

| CLI Flag | Environment Variable | Description |
|----------|---------------------|-------------|
| `--ssh_ids` | `MLC_MULTINODE_SYSTEM_SSH_IDS` | Comma-separated SSH targets (`user@host:port`) |
| `--system_name` | `MLC_MLPERF_SYSTEM_NAME` | System identifier (required) |
| `--config_file` | `MLC_MLPERF_CONFIG_FILE` | Path to a JSON / YAML config file |
| `--endpoint_url` | `MLC_MLPERF_ENDPOINT_URL` | Endpoint URL for serving framework auto-detection |
| `--submitter_org_names` | `MLC_MLPERF_SUBMITTER` | Submitting organization name |
| `--division` | `MLC_MLPERF_SUBMISSION_DIVISION` | `open` or `closed` |
| `--category` | `MLC_MLPERF_SUBMISSION_SYSTEM_TYPE` | `datacenter` or `edge` |
| `--system_availability_status` | `MLC_MLPERF_SUBMISSION_SYSTEM_STATUS` | `available`, `preview`, or `rdi` |
| `--cooling` | `MLC_MLPERF_COOLING` | Cooling method |
| `--hw_notes` | `MLC_MLPERF_HARDWARE_NOTES` | Hardware notes |
| `--system_type_detail` | `MLC_MLPERF_SYSTEM_TYPE_DETAIL` | `cloud`, `on-premise`, `edge-server`, or `edge-device` (optional) |

If you hit any issues while using this script, please feel free to raise an issue at [https://github.com/mlcommons/mlperf-automations](https://github.com/mlcommons/mlperf-automations).
