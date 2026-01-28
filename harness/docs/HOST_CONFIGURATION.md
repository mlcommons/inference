# Host Configuration in YAML and Command Line

## Overview

The harness supports specifying API server hosts in YAML configuration files, with optional command-line overrides. This allows you to:
- Configure hosts in YAML for reusable configurations
- Override hosts on the command line for flexibility
- Support single host or multiple hosts for load balancing
- Combine hosts with ports for flexible server addressing

## YAML Configuration

### Single Host

```yaml
backend: vllm
model: meta-llama/Llama-3.1-8B-Instruct
host: server1.example.com
port: 8000
```

### Multiple Hosts (Load Balancing)

```yaml
backend: vllm
model: meta-llama/Llama-3.1-8B-Instruct
host: [server1.example.com, server2.example.com, server3.example.com]
port: 8000
```

### Multiple Hosts with Multiple Ports

```yaml
backend: vllm
model: meta-llama/Llama-3.1-8B-Instruct
host: [server1.example.com, server2.example.com, server3.example.com]
ports: [8000, 8001, 8002]
```

**Note**: If using multiple hosts with multiple ports, the number of hosts must match the number of ports.

### Single Host with Multiple Ports

```yaml
backend: vllm
model: meta-llama/Llama-3.1-8B-Instruct
host: server1.example.com
ports: [8000, 8001, 8002]
```

## Command Line Override

### Override Single Host

```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --test-mode performance \
    --server-config configs/backends/vllm.yaml \
    --host server2.example.com \
    --output-dir ./harness_output
```

### Override with Multiple Hosts (Load Balancing)

```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --test-mode performance \
    --server-config configs/backends/vllm.yaml \
    --host server1.example.com,server2.example.com,server3.example.com \
    --output-dir ./harness_output
```

## Priority Order

The host is determined in the following priority order (highest to lowest):

1. **Command line `--host`**: Always takes precedence
2. **YAML config `host`**: Used if no command line override
3. **Default**: `localhost` (if server is started by harness)

## Examples

### Example 1: YAML Configuration Only

**config.yaml**:
```yaml
backend: vllm
model: meta-llama/Llama-3.1-8B-Instruct
host: inference-server.example.com
port: 8000
```

**Command**:
```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --test-mode performance \
    --server-config config.yaml \
    --output-dir ./harness_output
```

**Result**: Connects to `http://inference-server.example.com:8000`

### Example 2: Command Line Override

**config.yaml**:
```yaml
backend: vllm
model: meta-llama/Llama-3.1-8B-Instruct
host: server1.example.com
port: 8000
```

**Command**:
```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --test-mode performance \
    --server-config config.yaml \
    --host server2.example.com \
    --output-dir ./harness_output
```

**Result**: Connects to `http://server2.example.com:8000` (YAML host is overridden)

### Example 3: Load Balancing with Multiple Hosts

**config.yaml**:
```yaml
backend: vllm
model: meta-llama/Llama-3.1-8B-Instruct
host: [server1.example.com, server2.example.com, server3.example.com]
port: 8000
```

**Command**:
```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --test-mode performance \
    --server-config config.yaml \
    --output-dir ./harness_output
```

**Result**: Load balances across:
- `http://server1.example.com:8000`
- `http://server2.example.com:8000`
- `http://server3.example.com:8000`

### Example 4: Command Line Override with Multiple Hosts

**Command**:
```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --test-mode performance \
    --server-config config.yaml \
    --host prod-server1.example.com,prod-server2.example.com \
    --output-dir ./harness_output
```

**Result**: Load balances across the command-line specified hosts (YAML hosts are overridden)

## Integration with API Server URL

The host configuration works alongside the `--api-server-url` option:

- **If `--api-server-url` is specified**: It takes precedence over host/port configuration
- **If `--host` is specified**: Constructs URL from host + port (from YAML or default 8000)
- **If neither is specified**: Uses host from YAML config, or `localhost` if server is started by harness

### Example: Using --api-server-url (Takes Precedence)

```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --test-mode performance \
    --server-config config.yaml \
    --api-server-url http://custom-server.example.com:9000 \
    --output-dir ./harness_output
```

**Result**: Uses `http://custom-server.example.com:9000` (ignores host/port from YAML)

## Server Startup

When the harness starts a server (if `--api-server-url` is not provided), it uses:

1. **Host from YAML config** (if specified)
2. **Default**: `localhost`

The server will bind to the specified host (or `localhost` by default).

## Best Practices

1. **Use YAML for Defaults**: Configure common hosts in YAML for team-wide consistency
2. **Use Command Line for Overrides**: Override hosts on command line for testing or different environments
3. **Document Hosts**: Include host information in YAML comments for clarity
4. **Network Configuration**: Ensure hosts are accessible from the harness machine

## Troubleshooting

### Issue: "Connection refused" or "Cannot connect to host"
- **Solution**: Verify the host is accessible from your machine
- Check network connectivity: `ping <host>`
- Verify firewall rules allow connections to the port
- Ensure the server is running on the specified host

### Issue: "Number of hosts must match number of ports"
- **Solution**: When using multiple hosts with multiple ports, ensure:
  - `host` is a list with N elements
  - `ports` is a list with N elements
  - Both lists have the same length

### Issue: Host from YAML not being used
- **Solution**: Check if `--api-server-url` or `--host` is specified (they override YAML)
- Verify YAML syntax is correct (host should be a string or list)
- Check that `--server-config` points to the correct YAML file

## Technical Details

### URL Construction

The API server URL is constructed as:
```
http://<host>:<port>
```

Where:
- `<host>`: From `--host`, YAML `host`, or `localhost` (default)
- `<port>`: From YAML `port` or `8000` (default)

### Load Balancing

When multiple hosts are specified:
- URLs are constructed for each host+port combination
- Load balancing is handled by the `LoadGenClient`
- See [LOAD_BALANCING.md](LOAD_BALANCING.md) for details
