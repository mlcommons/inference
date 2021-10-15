#!/bin/bash

output_dir=${OUTPUT_DIR:-./}
system_id=$1

# Create output folder
if [ ! -d "${output_dir}/systems" ]; then
    mkdir ${output_dir}/systems
fi

# Get system information
core_count=$(grep -c ^processor /proc/cpuinfo)
memory_capacity=$(free -h | awk  '/Mem:/{print $2}')
cpu_name=$(lscpu | grep 'Model name' | cut -f 2 -d ":" | awk '{$1=$1}1')
processor_frequency=${cpu_name#*@}
storage_capacity=$(df -h | grep "/$" | awk 'END{print $2}')

if [ "$(cat /sys/block/sda/queue/rotational)" = "1" ]; then
  storage_type="HDD"
else
  storage_type="SDD"
fi

# Save system specs in output file
cat > ${output_dir}/systems/${system_id}.json <<EOF
{
        "division": "closed",
        "status": "available",
        "submitter": "mlperf-org",
        "system_name": "${system_id}",
        "system_type": "datacenter",
        
        "number_of_nodes": 1,
        "host_memory_capacity": "${memory_capacity}",
        "host_processor_core_count": ${core_count},
        "host_processor_frequency": "${processor_frequency}",
        "host_processor_model_name": "${cpu_name}",
        "host_processors_per_node": 1,
        "host_storage_capacity": "${storage_capacity}",
        "host_storage_type": "${storage_type}",
        
        "accelerator_frequency": "-",
        "accelerator_host_interconnect": "-",
        "accelerator_interconnect": "-",
        "accelerator_interconnect_topology": "-",
        "accelerator_memory_capacity": "16GB",
        "accelerator_memory_configuration": "none",
        "accelerator_model_name": "T4",
        "accelerator_on-chip_memories": "-",
        "accelerators_per_node": 1,

        "framework": "v1.14.0-rc1-22-gaf24dc9",
        "operating_system": "ubuntu-18.04",
        "other_software_stack": "cuda-11.2",
        "sw_notes": ""
}
EOF