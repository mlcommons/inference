# Load Balancing Support for Multiple API Servers

## Overview

The harness now supports load balancing across multiple API server URLs. This allows you to:
- Distribute requests across multiple inference servers
- Improve throughput and fault tolerance
- Automatically retry failed requests on alternate servers
- Use round-robin or random load balancing strategies

## Usage

### Command Line Options

You can specify multiple API server URLs in two ways:

#### Option 1: Comma-Separated URLs
```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://server1:8000,http://server2:8000,http://server3:8000 \
    --output-dir ./harness_output
```

#### Option 2: Multiple --api-server-urls Flags
```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --test-mode performance \
    --api-server-urls http://server1:8000 http://server2:8000 http://server3:8000 \
    --output-dir ./harness_output
```

### Load Balancing Strategies

The harness supports two load balancing strategies:

1. **Round-Robin** (default): Distributes requests sequentially across servers
2. **Random**: Randomly selects a server for each request

The strategy can be configured via the `load_balance_strategy` parameter in the server config:

```yaml
backend: vllm
config:
  load_balance_strategy: round_robin  # or 'random'
```

### Retry Logic

When a request fails (connection error or non-200 status code), the harness will:
1. Automatically retry on the next available server
2. Continue trying servers until one succeeds or all are exhausted
3. Log warnings for failed servers

### Examples

#### Basic Load Balancing
```bash
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:8000,http://localhost:8001,http://localhost:8002 \
    --output-dir ./harness_output
```

#### GPT-OSS-120B with Load Balancing
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://sglang1:30000,http://sglang2:30000,http://sglang3:30000 \
    --output-dir ./harness_output
```

#### Qwen3VL with Load Balancing
```bash
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://vllm1:8000,http://vllm2:8000,http://vllm3:8000 \
    --output-dir ./harness_output
```

#### Server Scenario with Load Balancing
```bash
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --scenario Server \
    --test-mode performance \
    --api-server-url http://server1:8000,http://server2:8000 \
    --server-target-qps 20.0 \
    --output-dir ./harness_output
```

## Configuration

### Server Config

You can configure load balancing behavior in the server config YAML:

```yaml
backend: vllm
model: meta-llama/Llama-3.1-8B-Instruct
config:
  load_balance_strategy: round_robin  # or 'random'
  max_retries_per_server: 3
```

### Parameters

- **load_balance_strategy**: `'round_robin'` (default) or `'random'`
- **max_retries_per_server**: Maximum retries per server before trying next (default: 3)

## How It Works

### Request Distribution

1. **Round-Robin**: Requests are distributed sequentially:
   - Request 1 → Server 1
   - Request 2 → Server 2
   - Request 3 → Server 3
   - Request 4 → Server 1 (wraps around)
   - ...

2. **Random**: Each request is sent to a randomly selected server

### Failure Handling

When a server fails:
1. The request is retried on the next available server
2. Failed servers are temporarily excluded from the pool
3. If all servers fail, the request fails with an error

### Health Checks

During initialization, the harness checks that at least one server is ready:
- Waits for servers to become available (up to 600 seconds timeout)
- Logs which servers are ready
- Excludes unavailable servers from the load balancing pool

## Best Practices

1. **Server Configuration**: Ensure all servers have the same model and configuration
2. **Network**: Use servers on the same network for consistent latency
3. **Monitoring**: Monitor individual server health and performance
4. **Capacity Planning**: Distribute load evenly across servers

## Limitations

- Load balancing is done at the request level (not batch level)
- All servers should have the same model and configuration
- No automatic server health recovery (failed servers remain excluded)
- Round-robin is per-request, not per-batch

## Troubleshooting

### Issue: "No API servers became ready"
- **Solution**: Check that at least one server is running and accessible
- Verify server URLs are correct
- Check network connectivity

### Issue: "All servers marked as failed"
- **Solution**: Check server logs for errors
- Verify servers are running and healthy
- Check network connectivity

### Issue: Uneven load distribution
- **Solution**: Use round-robin strategy (default)
- Ensure all servers have similar capacity
- Check for network latency differences

## Technical Details

### Implementation

- Load balancing is implemented in `LoadGenClient`
- Each request selects a server using `_get_next_server_url()`
- Retry logic is handled in `_send_request_with_retry()`
- Failed servers are tracked in `self.failed_servers`

### Thread Safety

- Round-robin uses a counter (`self.current_server_index`) that is thread-safe
- Random selection uses Python's `random.choice()` which is thread-safe
- Server selection is done per-request, not per-thread

### Performance

- Load balancing adds minimal overhead (< 1ms per request)
- Retry logic only activates on failures
- Health checks are done once during initialization
