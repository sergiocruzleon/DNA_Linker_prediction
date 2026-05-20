# DNA Linker Prediction Pipeline

A Python pipeline for predicting DNA linker connections between nucleosome particles from Cryo-ET tomograms.

## Overview

This pipeline identifies potential DNA linker connections between nucleosome particles based on:
- **Spatial proximity**: Distance-based filtering using configurable tracing distance
- **Directional compatibility**: Angular matching between particle entry/exit points
- **Energy minimization**: Persistence length-based bending energy calculation

## Features

- **Parallel processing**: Multi-worker support via joblib for handling multiple datasets
- **YAML-based configuration**: All parameters configurable via external YAML file
- **Per-project configs**: Use `--config` argument for different projects

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DNA_Linker_prediction.git
cd DNA_Linker_prediction

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install just the package
pip install -e .
```

## Quick Start

### Command Line Interface

```bash
# Run pipeline with default config
python scripts/run_pipeline.py --emd 2601 --suffix STA_tmpl

# Run on multiple datasets
python scripts/run_pipeline.py --emd 2601 13356 13363 --suffix STA_tmpl

# Run with parallel workers
python scripts/run_pipeline.py --emd 2601 --workers 4

# Use custom config for a specific project
python scripts/run_pipeline.py --config /path/to/project_config.yaml --emd 2601
```

### Python API

```python
from pathlib import Path
from cryocat import cryomotl
from dna_linker import config, run_pipeline

# Set up paths
input_dir = Path(config.input_dir)
output_dir = Path(config.output_base)

# Run the pipeline
run_pipeline.run_full_pipeline(
    path_mask=str(input_dir),
    motl_name="motl_EMD2601_STA_tmpl.em",
    entry=config.entry_mask,
    exit=config.exit_mask,
    origin_entry=config.origin_entry_mask,
    origin_exit=config.origin_exit_mask,
    path_output=str(output_dir),
    tracing_distance=config.tracing_distance,
    max_distance=config.tracing_distance / (config.pixel_size * config.bin),
)
```

## Configuration

### Using YAML Configuration File

All parameters can be configured in a YAML file. The default is `dna_linker/pipeline_config.yaml`:

```bash
# Use default config
python scripts/run_pipeline.py --emd 2601

# Use custom config file
python scripts/run_pipeline.py --config /path/to/my_project_config.yaml --emd 2601
```

### Creating a Custom Config File

Create a YAML file for your project:

```yaml
# my_project_config.yaml

# Physics Parameters
pixel_size: 1.0      # Voxel size in Armstrong
bin: 1.0           # Binning factor
lp: 500            # Persistence length (nm)
lo: 150            # Contour length (nm)
tracing_distance: 350  # Tracing distance in nm

# Processing
workers: 4
skip_tracing: false

# Input/Output
input_dir: "/path/to/your/input/data"
output_base: "/path/to/output/directory"

# Mask Filenames
entry_mask: "your_entry_mask.mrc"
exit_mask: "your_exit_mask.mrc"
origin_entry_mask: "your_origin_entry_mask.mrc"
origin_exit_mask: "your_origin_exit_mask.mrc"

# File Patterns (use {emd_id}, {suffix}, {tracing_distance} as placeholders)
motl_pattern: "motl_EMD{emd_id}_{suffix}.em"
traced_pattern: "EMD{emd_id}_tr{tracing_distance}nm_{suffix}.em"
output_tomo_dir: "EMD{emd_id}_{suffix}/"
output_clusters: "outputs_EMD{emd_id}_{suffix}/clusters/"
output_linkers: "outputs_EMD{emd_id}_{suffix}/linkers/"
output_dictionary: "outputs_EMD{emd_id}_{suffix}/dictionary/"

# Default suffix
suffix: "my_suffix"
```

### Configuration Parameters

#### Physics Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pixel_size` | 1.0 | Voxel size in Armstrong |
| `bin` | 1.0 | Binning factor |
| `lp` | 500 | Persistence length in nm (before binning) |
| `lo` | 150 | Contour length in nm (before binning) |
| `tracing_distance` | 350 | Maximum tracing distance in nm |

#### Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `workers` | 4 | Number of parallel workers |
| `skip_tracing` | false | Skip tracing step if output exists |

#### Input/Output Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dir` | ./dna_linker/inputs | Input directory |
| `output_base` | ./dna_linker/outputs | Output base directory |

#### Mask Filenames

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entry_mask` | Threshold_ref_entrymask_r2_resamp_righthand.mrc | Entry mask filename |
| `exit_mask` | Threshold_ref_exitmask_r2_resamp_righthand.mrc | Exit mask filename |
| `origin_entry_mask` | Threshold_ref_Origin_entrymask_r2_resamp_righthand.mrc | Origin entry mask |
| `origin_exit_mask` | Threshold_ref_Origin_exitmask_r2_resamp_righthand.mrc | Origin exit mask |

#### Path Patterns

| Parameter | Default | Description |
|-----------|---------|-------------|
| `motl_pattern` | motl_EMD{emd_id}_{suffix}.em | MOTL file name pattern |
| `traced_pattern` | EMD{emd_id}_tr{tracing_distance}nm_{suffix}.em | Traced file pattern |
| `output_tomo_dir` | EMD{emd_id}_{suffix}/ | Output tomogram directory |
| `output_clusters` | outputs_EMD{emd_id}_{suffix}/clusters_20nm/ | Clusters output path |
| `output_linkers` | outputs_EMD{emd_id}_{suffix}/A_linkers_20nm/ | Linkers output path |
| `output_dictionary` | outputs_EMD{emd_id}_{suffix}/A_Connections_dictionary_20nm/ | Dictionary output path |

### Programmatic Access

You can also access config values in Python:

```python
from dna_linker import config

# Physics parameters
print(config.lp)  # Persistence length
print(config.lo)  # Contour length
print(config.tracing_distance)
print(config.pixel_size)
print(config.bin)

# Pipeline settings
print(config.input_dir)
print(config.entry_mask)
print(config.workers)

# Path patterns
print(config.motl_pattern)
print(config.output_clusters)
```

Or load a custom config file:

```python
from dna_linker.config import get_config_for_run

# Load custom config
cfg = get_config_for_run("/path/to/custom_config.yaml")
print(cfg.input_dir)
print(cfg.motl_pattern)
```

## GPU Acceleration

The pipeline supports GPU acceleration for probability calculations using PyTorch. This can significantly speed up processing for large datasets.

### Supported GPUs

- **Apple Silicon** (M1/M2/M3 chips) via Metal Performance Shaders (MPS)
- **NVIDIA GPUs** via CUDA

### Batched GPU Processing

The pipeline now supports **batched GPU processing**, which sends multiple clusters to the GPU at once for maximum throughput. This is enabled by default when `gpu_accelerate: true`.

```yaml
# my_project_config.yaml

# Processing
gpu_accelerate: true     # Enable GPU acceleration
batch_gpu: true         # Batch multiple clusters on GPU (default: true)
```

To disable batched processing and use sequential processing:

```yaml
batch_gpu: false
```

### Enabling GPU Acceleration

To enable GPU acceleration, set `gpu_accelerate: true` in your YAML config file:

```yaml
# my_project_config.yaml

# Processing
gpu_accelerate: true
```

Or use the Python API:

```python
from dna_linker import config

# Enable GPU acceleration
config.gpu_accelerate = True

# Check GPU status
from dna_linker.dna_linkers import get_gpu_info
print(get_gpu_info())
# Output: {'device': 'mps', 'mps_available': True, 'cuda_available': False}
```

### Checking GPU Availability

```python
from dna_linker.dna_linkers import get_gpu_info

info = get_gpu_info()
print(f"PyTorch available: {info['torch_available']}")
print(f"Device: {info['device']}")  # 'cuda', 'mps', or 'cpu'
print(f"CUDA available: {info['cuda_available']}")
print(f"MPS available: {info['mps_available']}")
```

### Installation

PyTorch is included as a dependency. For best performance on Apple Silicon:

```bash
# Install PyTorch with MPS support (recommended for Mac)
pip install torch
```

For NVIDIA GPUs:

```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Testing

```bash
# Run unit tests
pytest tests/test_core_functions.py -v

# Run integration tests
pytest tests/test_regression.py -v --timeout=300

# Run benchmarks
python benchmarks/benchmark_core.py --scaling
```

## Project Structure

```
DNA_Linker_prediction/
├── dna_linker/              # Main Python package
│   ├── __init__.py
│   ├── config.py             # Configuration loader
│   ├── pipeline_config.yaml  # Default YAML configuration
│   ├── dna_linkers.py       # Core probability and connection logic
│   ├── utils_motlFiles.py   # MOTL file utilities
│   └── ...
├── tests/                    # Test suite
├── benchmarks/               # Performance benchmarks
├── scripts/                  # Command-line scripts
├── notebooks/                # Example Jupyter notebooks
└── docs/                    # Documentation
```

## Documentation

See the `docs/` directory for detailed documentation:
- `scalability_roadmap.md` - Performance optimization roadmap

## License

MIT License

## Authors

- Sergio Cruz
