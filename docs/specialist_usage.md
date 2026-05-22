# Specialist Usage Notes

This page collects configuration details that are useful for adapting the pipeline to a new dataset.

## Configuration

The default configuration file is `dna_linker/pipeline_config.yaml`.

```yaml
pixel_size: 1.0
bin: 1.0
lp: 500
lo: 150
tracing_distance: 350

workers: 4
skip_tracing: false
gpu_accelerate: false
batch_gpu: true
save_plots: true

input_dir: "./dna_linker/inputs"
output_base: "./dna_linker/outputs"
motl_file: "my_particles.em"

entry_mask: "Threshold_ref_entrymask_r2_resamp_righthand.mrc"
exit_mask: "Threshold_ref_exitmask_r2_resamp_righthand.mrc"
origin_entry_mask: "Threshold_ref_Origin_entrymask_r2_resamp_righthand.mrc"
origin_exit_mask: "Threshold_ref_Origin_exitmask_r2_resamp_righthand.mrc"
```

`motl_file` is optional. If it is unset, the command-line runner falls back to:

```yaml
motl_pattern: "motl_EMD{emd_id}_{suffix}.em"
```

Relative MOTL filenames are resolved from `input_dir`; absolute paths are accepted.

## Command-Line Examples

Run the included example:

```bash
python scripts/run_pipeline.py --emd 2601 --motl-file motl_EMD2601_dropped_01.em --workers 1
```

Install notebook tooling for the example notebook:

```bash
pip install -e ".[examples]"
```

Run a project-specific config:

```bash
python scripts/run_pipeline.py --config /path/to/project_config.yaml --emd 2601 --motl-file your_particles.em
```

Estimate runtime and memory without running the full pipeline:

```bash
python scripts/run_pipeline.py --estimate --emd 2601 --motl-file your_particles.em
```

## Python API

```python
from pathlib import Path
from dna_linker import config, run_pipeline

input_dir = Path(config.input_dir)
output_dir = Path(config.output_base)
output_dir.mkdir(parents=True, exist_ok=True)

run_pipeline.run_full_pipeline(
    path_mask=str(input_dir) + "/",
    motl_name="motl_EMD2601_dropped_01.em",
    entry=config.entry_mask,
    exit=config.exit_mask,
    origin_entry=config.origin_entry_mask,
    origin_exit=config.origin_exit_mask,
    path_output=str(output_dir) + "/",
    motl_trace_input=str(output_dir / "EMD2601_tr350nm_example.em"),
    tracing_distance=config.tracing_distance,
    max_distance=config.tracing_distance / (config.pixel_size * config.bin),
    output_path_cluster=str(output_dir / "clusters_20nm") + "/",
    output_path_linker=str(output_dir / "A_linkers_20nm") + "/",
    output_path_dictionary=str(output_dir / "A_Connections_dictionary_20nm") + "/",
    dnal_object=config.lo,
    lp_object=config.lp,
    max_processes=1,
)
```

## GPU Notes

GPU acceleration uses PyTorch when available. Set this in the YAML config:

```yaml
gpu_accelerate: true
batch_gpu: true
```

Check GPU availability from Python:

```python
from dna_linker.dna_linkers import get_gpu_info

print(get_gpu_info())
```

## Repository Scope

This public release keeps the runnable example and removes internal benchmark and test folders. Generated outputs are intentionally ignored by Git.

## Contact

Sergio Cruz: sn.cruz35@gmail.com
