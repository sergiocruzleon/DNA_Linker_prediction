# DNA Linker Prediction Pipeline

A high-performance Python pipeline for predicting DNA linker connections between nucleosome particles in Cryo-EM tomograms.

## Overview

This pipeline identifies potential DNA linker connections between nucleosome particles based on:
- **Spatial proximity**: Distance-based filtering using configurable tracing distance
- **Directional compatibility**: Angular matching between particle entry/exit points
- **Energy minimization**: Persistence length-based bending energy calculation

## Features

- **Vectorized probability computation**: NumPy broadcasting for efficient N×N×4 probability calculations
- **Parallel processing**: Multi-worker support via joblib for handling multiple datasets
- **Modular design**: Separate components for configuration, probability calculation, and output generation
- **Comprehensive testing**: Unit tests and integration regression tests
- **CI/CD**: GitHub Actions workflow for automated testing and benchmarking

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dna_linker.git
cd dna_linker

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install just the package
pip install -e .
```

## Quick Start

### Command Line Interface

```bash
# Run pipeline on single dataset
python scripts/run_pipeline.py --emd 2601 --suffix STA_tmpl

# Run on multiple datasets
python scripts/run_pipeline.py --emd 2601 13356 13363 --suffix STA_tmpl

# Run with parallel workers
python scripts/run_pipeline.py --emd 2601 --workers 4
```

### Python API

```python
from pathlib import Path
from cryocat import cryomotl
from dna_linker import config, run_pipeline

# Set up paths
input_dir = Path("dna_linker/inputs")
output_dir = Path("dna_linker/outputs")

# Run the pipeline
run_pipeline.run_full_pipeline(
    path_mask=str(input_dir),
    motl_name="motl_EMD2601_STA_tmpl.em",
    entry="Threshold_ref_entrymask_r2_resamp_righthand.mrc",
    exit="Threshold_ref_exitmask_r2_resamp_righthand.mrc",
    origin_entry="Threshold_ref_Origin_entrymask_r2_resamp_righthand.mrc",
    origin_exit="Threshold_ref_Origin_exitmask_r2_resamp_righthand.mrc",
    path_output=str(output_dir),
    tracing_distance=350,
    max_distance=350,
)
```

## Configuration

Key parameters in `dna_linker/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tracing_distance` | 350 | Maximum tracing distance in voxels |
| `pixel_size` | 1.0 | Voxel size in nm |
| `bin` | 1.0 | Binning factor |
| `lo` | 50.0 | Persistence length in voxels |
| `pmin` | 0.1 | Minimum probability threshold |

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
dna_linker/
├── dna_linker/           # Main Python package
│   ├── __init__.py
│   ├── config.py         # Configuration parameters
│   ├── dna_linkers.py    # Core probability and connection logic
│   ├── utils_motlFiles.py # MOTL file utilities
│   └── ...
├── tests/                # Test suite
├── benchmarks/           # Performance benchmarks
├── scripts/              # Command-line scripts
├── notebooks/            # Example Jupyter notebooks
└── docs/                 # Documentation
```

## Documentation

See the `docs/` directory for detailed documentation:
- `scalability_roadmap.md` - Performance optimization roadmap

## License

MIT License

## Authors

- Sergio Cruz
