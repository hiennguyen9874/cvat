# GPU Allocation Strategy for Nuclio Workers

## Overview
This configuration implements a GPU allocation strategy where each worker gets assigned to a specific GPU device using the modulo operation: `device_id = worker_id % num_gpus`.

## Current Configuration

### Function Configuration (owlv2/function-gpu.yaml)
- **Workers**: 4 (`maxWorkers: 4`)
- **Total GPUs**: 4 (`nvidia.com/gpu: 4`)
- **GPU per Worker**: 1 (distributed via modulo)

### GPU Assignment Logic (owlv2/main.py)
```python
# Handle worker_id type conversion (Nuclio passes it as string)
worker_id_raw = getattr(context, 'worker_id', 0)
if isinstance(worker_id_raw, str):
    worker_id = int(worker_id_raw)
else:
    worker_id = worker_id_raw

num_gpus = torch.cuda.device_count()
device_id = worker_id % num_gpus
device = torch.device(f"cuda:{device_id}")
```

## How It Works

### With 4 Workers and 4 GPUs:
- **Worker 0**: `0 % 4 = 0` → Uses `cuda:0`
- **Worker 1**: `1 % 4 = 1` → Uses `cuda:1`
- **Worker 2**: `2 % 4 = 2` → Uses `cuda:2`
- **Worker 3**: `3 % 4 = 3` → Uses `cuda:3`

### With 4 Workers and 2 GPUs:
- **Worker 0**: `0 % 2 = 0` → Uses `cuda:0`
- **Worker 1**: `1 % 2 = 1` → Uses `cuda:1`
- **Worker 2**: `2 % 2 = 0` → Uses `cuda:0` (shared)
- **Worker 3**: `3 % 2 = 1` → Uses `cuda:1` (shared)

### With 2 Workers and 4 GPUs:
- **Worker 0**: `0 % 4 = 0` → Uses `cuda:0`
- **Worker 1**: `1 % 4 = 1` → Uses `cuda:1`
- GPUs 2 and 3 remain available for additional workers

## Bug Fix Applied

**Issue**: `TypeError: not all arguments converted during string formatting`
- **Cause**: Nuclio passes `worker_id` as a string, but modulo operator needs integer
- **Solution**: Added type conversion to handle both string and integer worker_id

```python
# Before (caused error)
worker_id = getattr(context, 'worker_id', 0)
device_id = worker_id % num_gpus  # Error: string % int

# After (fixed)
worker_id_raw = getattr(context, 'worker_id', 0)
worker_id = int(worker_id_raw) if isinstance(worker_id_raw, str) else worker_id_raw
device_id = worker_id % num_gpus  # Works: int % int
```

## Benefits

1. **Automatic Load Distribution**: Workers are automatically distributed across available GPUs
2. **Scalable**: Works with any number of workers and GPUs
3. **Fault Tolerant**: Fallback to CPU if no GPUs available
4. **Resource Efficient**: Optimal GPU utilization
5. **Type Safe**: Handles both string and integer worker_id

## Deployment

```bash
# Deploy the function
DOCKER_BUILDKIT=1 ./serverless/deploy_gpu.sh owlv2

# Verify GPU allocation in logs
nuctl get logs owlv2 --namespace nuclio
```

## Monitoring

Check GPU usage across workers:
```bash
# View GPU utilization
kubectl exec -it <owlv2-pod> -- nvidia-smi

# Check worker assignments in logs
kubectl logs <owlv2-pod> | grep "Worker.*assigned to GPU"
```

## Example Log Output (Fixed)
```
Worker ID: 0
Number of available GPUs: 4
Worker 0 assigned to GPU device: cuda:0

Worker ID: 1
Number of available GPUs: 4
Worker 1 assigned to GPU device: cuda:1

Worker ID: 2
Number of available GPUs: 4
Worker 2 assigned to GPU device: cuda:2

Worker ID: 3
Number of available GPUs: 4
Worker 3 assigned to GPU device: cuda:3
``` 