# Scalability Roadmap

This document outlines the path forward for scaling the DNA Linker prediction pipeline to handle tens of thousands of particles efficiently.

## Current Status

After optimization #1 (vectorization), the probability matrix calculation now runs in ~2-3ms regardless of particle count. The main computational bottlenecks have shifted:

| Stage | Current Performance | Notes |
|-------|---------------------|-------|
| I/O (MRC/EM files) | Varies | Disk-bound |
| Distance matrix | O(N²) NumPy | Fast for N < 1000 |
| Probability matrix | ~3ms (vectorized) | No longer a bottleneck |
| Graph construction | Unknown | Likely next target |
| Clustering | Unknown | Likely next target |

## Scaling Targets

| Particle Count | Target Runtime | Priority |
|----------------|----------------|----------|
| 100 | < 1 second | High |
| 1,000 | < 30 seconds | High |
| 10,000 | < 5 minutes | Medium |
| 100,000 | < 30 minutes | Future |

---

## Phase 1: Multi-core CPU Parallelization

### When to Use
- Particle counts: 1,000 - 10,000
- Available cores: 4+ cores
- Memory: 16GB+ RAM

### Implementation Strategy

#### Option A: joblib for Particle-wise Parallelism

```python
from joblib import Parallel, delayed
import numpy as np

def compute_particle_stats(particle_data):
    """Compute statistics for a single particle."""
    # Independent particle processing
    return result

# Use all cores for particle-wise computations
results = Parallel(n_jobs=-1)(
    delayed(compute_particle_stats)(p) for p in particles
)
```

**Risk**: Low  
**Expected speedup**: 2-4x on 4-8 cores  
**Complexity**: Low  
**Correctness risk**: None (embarrassingly parallel)

#### Option B: Chunked Pairwise Computations

For N > 1000, the distance/probability matrices become large (N² floats = 8*N² bytes):

| N | Memory for N² float64 | Feasible? |
|---|----------------------|-----------|
| 100 | 80 KB | ✅ Yes |
| 1,000 | 80 MB | ✅ Yes |
| 10,000 | 8 GB | ⚠️ Large |
| 100,000 | 800 GB | ❌ No |

For large N, use chunked processing:

```python
def chunked_distance_matrix(positions, chunk_size=1000):
    """Compute distance matrix in chunks to limit memory."""
    n = len(positions)
    for i in range(0, n, chunk_size):
        chunk = positions[i:i+chunk_size]
        # Compute distances to all other particles
        yield compute_distances_chunk(chunk, positions)
```

**Risk**: Low  
**Memory reduction**: ~Nx chunk_size instead of N²  
**Complexity**: Medium  
**Correctness risk**: None (mathematically equivalent)

---

## Phase 2: GPU Acceleration

### When to Consider GPU
- Particle counts: 10,000+
- Need < 1 minute runtime
- GPU available (NVIDIA CUDA)

### Backend Options

| Backend | Pros | Cons | Best For |
|---------|------|------|----------|
| **CuPy** | NumPy-like API, easy migration | Requires CUDA | Drop-in replacement |
| **Numba CUDA** | Fine-grained control | Lower level | Custom kernels |
| **PyTorch** | Large ecosystem, autograd | Larger dependency | Neural network integration |
| **JAX** | Functional, JIT, vmap | Learning curve | Research/prototyping |

### CuPy Implementation (Recommended First)

CuPy provides a NumPy-compatible API that runs on GPU:

```python
import cupy as cp

def gpu_distance_matrix(positions):
    """GPU-accelerated distance matrix."""
    # Transfer to GPU
    pos_gpu = cp.asarray(positions)
    
    # GPU computation (thousands of cores!)
    diff = pos_gpu[:, cp.newaxis, :] - pos_gpu[cp.newaxis, :, :]
    dist = cp.sqrt(cp.sum(diff**2, axis=-1))
    
    # Transfer back to CPU
    return cp.asnumpy(dist)
```

**Expected speedup for N=10,000**:  
- CPU (NumPy): ~1 second  
- GPU (CuPy): ~10-50 ms  
- **Speedup: 20-100x**

### Memory Considerations

GPU memory is limited (typically 8-24 GB):

| N | Float32 matrix size | Feasible on 8GB GPU? |
|---|---------------------|----------------------|
| 10,000 | 400 MB | ✅ Yes |
| 50,000 | 10 GB | ⚠️ Borderline |
| 100,000 | 40 GB | ❌ No |

For very large datasets, combine GPU + chunking.

### Implementation Complexity

1. **Low risk**: Wrap existing functions with CuPy
2. **Medium complexity**: 2-4 hours for initial implementation
3. **Testing**: Need GPU for CI (can skip on some runs)

---

## Phase 3: Cluster/HPC Scaling

### When to Use Cluster
- Particle counts: 100,000+
- Need < 30 minute runtime
- Available HPC resources (SLURM, PBS, etc.)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Master Node                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │  Scheduler  │  │  I/O Server  │  │  Aggregator │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │ Worker  │      │ Worker  │      │ Worker  │
    │ Node 1  │      │ Node 2  │      │ Node N  │
    └─────────┘      └─────────┘      └─────────┘
```

### SLURM Job Script Example

```bash
#!/bin/bash
#SBATCH --job-name=dna_linker
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

# Load modules
module load python/3.11 cuda/12.1

# Run with chunking
python run_pipeline.py \
    --input /scratch/dataset \
    --output /scratch/output \
    --n-chunks 32 \
    --backend parallel
```

### Chunking Strategy

For 100,000+ particles, use hierarchical chunking:

1. **Spatial chunking**: Split particles by location (tiles)
2. **Temporal chunking**: Process in time-ordered batches
3. **Hybrid**: Combine both for optimal load balancing

```python
def hierarchical_processing(particles, n_spatial=4, n_temporal=8):
    """Process particles with spatial + temporal chunking."""
    spatial_chunks = spatial_partition(particles, n_spatial)
    
    for schunk in spatial_chunks:
        temporal_chunks = temporal_partition(schunk, n_temporal)
        
        for tchunk in temporal_chunks:
            yield process_chunk(tchunk)
```

### I/O Strategy

For large datasets:

1. **Memory-mapped files**: Use `numpy.memmap` for I/O
2. **Parallel HDF5**: Use `h5py` with MPI for concurrent reads
3. **Avoid per-particle disk hits**: Batch all reads/writes

```python
import h5py

def parallel_h5_read(filename, dataset_name, chunk_size=10000):
    """Read HDF5 dataset in chunks."""
    with h5py.File(filename, 'r') as f:
        dset = f[dataset_name]
        n = len(dset)
        
        for i in range(0, n, chunk_size):
            yield dset[i:i+chunk_size]
```

### Cluster Performance Targets

| Nodes | Cores | N=100,000 | N=1,000,000 |
|-------|-------|-----------|-------------|
| 1 | 8 | ~10 min | ~2 hours |
| 4 | 32 | ~3 min | ~30 min |
| 16 | 128 | ~1 min | ~8 min |

---

## Recommended Implementation Order

1. **Immediate**: Add joblib parallelization for particle-wise operations
2. **Short-term**: Implement chunked processing for N > 1000
3. **Medium-term**: Add CuPy GPU backend (opt-in flag)
4. **Long-term**: Add SLURM cluster support with chunking

### Backward Compatibility

All new backends will be opt-in:

```python
# Default: CPU (vectorized)
pipeline = DnaLinkerPipeline()

# Optional: Parallel CPU
pipeline = DnaLinkerPipeline(backend='parallel', n_jobs=8)

# Optional: GPU (requires CuPy)
pipeline = DnaLinkerPipeline(backend='gpu')

# Optional: Cluster (future)
pipeline = DnaLinkerPipeline(backend='cluster', n_nodes=4)
```

---

## Verification Strategy

Each scaling option requires:

1. **Correctness test**: Run on small dataset (N=50), compare with baseline
2. **Scaling test**: Verify linear scaling with particle count
3. **Memory test**: Monitor peak memory usage
4. **Integration test**: Full pipeline on regression dataset

---

## Dependencies Matrix

| Feature | Required Dependencies | Optional |
|---------|----------------------|----------|
| CPU (baseline) | numpy, pandas, scipy | - |
| Parallel CPU | numpy, pandas, scipy, joblib | ✅ Optional |
| GPU (CuPy) | numpy, cupy-cuda12x | ✅ Optional |
| Cluster | numpy, dask, mpi4py | ✅ Optional |
