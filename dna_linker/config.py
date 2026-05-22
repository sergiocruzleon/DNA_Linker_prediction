import yaml
from pathlib import Path

# Use non-interactive backend for matplotlib to avoid popup windows
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Default physics constants (used if not in YAML config)
DEFAULT_KB = 1.380649 * 10**(-23)  # J/K
DEFAULT_T = 300  # Kelvin (27 deg C)

# Load the repository example configuration by default.
_repo_root = Path(__file__).resolve().parent.parent
_config_path = _repo_root / "example" / "pipeline_config.yaml"
if _config_path.exists():
    with open(_config_path, 'r') as f:
        _yaml_config = yaml.safe_load(f)
else:
    _yaml_config = {}

# Physics parameters - from YAML or defaults
# These can be customized per project in a YAML config file
pixel_size = _yaml_config.get('pixel_size', 1.0)  # Armstrong/voxel
bin = _yaml_config.get('bin', 1.0)  # Binning factor
lp = _yaml_config.get('lp', 500) / (bin*pixel_size)  # Persistence length (in voxels)
lo = _yaml_config.get('lo', 150) /  (bin*pixel_size)  # Contour length (in voxels)
tracing_distance = _yaml_config.get('tracing_distance', 350)  # nm

# Derived constants
kB = DEFAULT_KB
T = DEFAULT_T
pmin = 0.1

# Pipeline settings from YAML
input_dir = _yaml_config.get('input_dir', './example/inputs')
output_base = _yaml_config.get('output_base', './example/outputs')
motl_file = _yaml_config.get('motl_file')
entry_mask = _yaml_config.get('entry_mask', 'Threshold_ref_entrymask_r2_resamp_righthand.mrc')
exit_mask = _yaml_config.get('exit_mask', 'Threshold_ref_exitmask_r2_resamp_righthand.mrc')
origin_entry_mask = _yaml_config.get('origin_entry_mask', 'Threshold_ref_Origin_entrymask_r2_resamp_righthand.mrc')
origin_exit_mask = _yaml_config.get('origin_exit_mask', 'Threshold_ref_Origin_exitmask_r2_resamp_righthand.mrc')
suffix = _yaml_config.get('suffix', 'STA_templ')
workers = _yaml_config.get('workers', 4)
skip_tracing = _yaml_config.get('skip_tracing', False)
gpu_accelerate = _yaml_config.get('gpu_accelerate', False)
single_cluster_file = _yaml_config.get('single_cluster_file', False)
batch_gpu = _yaml_config.get('batch_gpu', True)  # Batch multiple clusters on GPU
save_plots = _yaml_config.get('save_plots', False)  # Save connectivity plots (blocks if True)

# Path patterns from YAML
motl_pattern = _yaml_config.get('motl_pattern', 'motl_EMD{emd_id}_{suffix}.em')
traced_pattern = _yaml_config.get('traced_pattern', 'EMD{emd_id}_tr{tracing_distance}nm_{suffix}.em')
output_tomo_dir = _yaml_config.get('output_tomo_dir', 'EMD{emd_id}_{suffix}/')
output_clusters = _yaml_config.get('output_clusters', 'outputs_EMD{emd_id}_{suffix}/clusters_20nm/')
output_linkers = _yaml_config.get('output_linkers', 'outputs_EMD{emd_id}_{suffix}/A_linkers_20nm/')
output_dictionary = _yaml_config.get('output_dictionary', 'outputs_EMD{emd_id}_{suffix}/A_Connections_dictionary_20nm/')


def get_config_for_run(config_path: str = None) -> 'PipelineConfig':
    """Load configuration from a custom YAML file, or return default config.
    
    Args:
        config_path: Path to custom YAML config file. If None, uses example/pipeline_config.yaml
    
    Returns:
        PipelineConfig object with all settings
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        # Get the directory of the YAML file for relative paths
        yaml_dir = Path(config_path).parent.resolve()
    else:
        yaml_cfg = _yaml_config
        yaml_dir = _config_path.parent
    
    class PipelineConfig:
        pass
    
    cfg = PipelineConfig()
    
    # Store YAML dir for relative path resolution
    cfg._yaml_dir = yaml_dir
    
    # Physics
    cfg.pixel_size = yaml_cfg.get('pixel_size', 1.0)
    cfg.bin = yaml_cfg.get('bin', 1.0)
    cfg.lp = yaml_cfg.get('lp', 500) / (cfg.bin*cfg.pixel_size)
    cfg.lo = yaml_cfg.get('lo', 150) / (cfg.bin*cfg.pixel_size)
    cfg.tracing_distance = yaml_cfg.get('tracing_distance', 350)
    
    # Pipeline - resolve paths relative to YAML file
    raw_input_dir = yaml_cfg.get('input_dir', './example/inputs')
    raw_output_base = yaml_cfg.get('output_base', './example/outputs')
    
    # If path is not absolute, make it relative to YAML directory
    cfg.input_dir = str(yaml_dir / raw_input_dir) if not Path(raw_input_dir).is_absolute() else raw_input_dir
    cfg.output_base = str(yaml_dir / raw_output_base) if not Path(raw_output_base).is_absolute() else raw_output_base
    cfg.motl_file = yaml_cfg.get('motl_file')
    
    cfg.entry_mask = yaml_cfg.get('entry_mask', 'Threshold_ref_entrymask_r2_resamp_righthand.mrc')
    cfg.exit_mask = yaml_cfg.get('exit_mask', 'Threshold_ref_exitmask_r2_resamp_righthand.mrc')
    cfg.origin_entry_mask = yaml_cfg.get('origin_entry_mask', 'Threshold_ref_Origin_entrymask_r2_resamp_righthand.mrc')
    cfg.origin_exit_mask = yaml_cfg.get('origin_exit_mask', 'Threshold_ref_Origin_exitmask_r2_resamp_righthand.mrc')
    cfg.suffix = yaml_cfg.get('suffix', 'STA_templ')
    cfg.workers = yaml_cfg.get('workers', 8)
    cfg.skip_tracing = yaml_cfg.get('skip_tracing', False)
    cfg.gpu_accelerate = yaml_cfg.get('gpu_accelerate', False)
    cfg.single_cluster_file = yaml_cfg.get('single_cluster_file', False)
    cfg.batch_gpu = yaml_cfg.get('batch_gpu', True)  # Batch multiple clusters on GPU
    cfg.save_plots = yaml_cfg.get('save_plots', False)  # Save connectivity plots
    
    # Path patterns
    cfg.motl_pattern = yaml_cfg.get('motl_pattern', 'motl_EMD{emd_id}_{suffix}.em')
    cfg.traced_pattern = yaml_cfg.get('traced_pattern', 'EMD{emd_id}_tr{tracing_distance}nm_{suffix}.em')
    cfg.output_tomo_dir = yaml_cfg.get('output_tomo_dir', 'EMD{emd_id}_{suffix}/')
    cfg.output_clusters = yaml_cfg.get('output_clusters', 'outputs_EMD{emd_id}_{suffix}/clusters_20nm/')
    cfg.output_linkers = yaml_cfg.get('output_linkers', 'outputs_EMD{emd_id}_{suffix}/A_linkers_20nm/')
    cfg.output_dictionary = yaml_cfg.get('output_dictionary', 'outputs_EMD{emd_id}_{suffix}/A_Connections_dictionary_20nm/')
    
    # Single file option
    cfg.single_cluster_file = yaml_cfg.get('single_cluster_file', False)
    
    return cfg


def estimate_runtime(n_particles: int, n_clusters: int = 10, n_workers: int = 4) -> dict:
    """Estimate runtime and memory requirements for a run.
    
    Args:
        n_particles: Number of particles in the MOTL file
        n_clusters: Number of clusters (default: 10)
        n_workers: Number of parallel workers
    
    Returns:
        Dictionary with estimates for pairs, memory, and runtime
    """
    n_pairs = n_particles ** 2
    
    # Memory estimation (in GB)
    #_probs matrix: N x N x 4 x 8 bytes (float64)
    prob_matrix_gb = (n_pairs * 4 * 8) / 1e9
    # Positions arrays: 4 x N x 3 x 8 bytes
    positions_gb = (4 * n_particles * 3 * 8) / 1e9
    # Other overhead
    overhead_gb = 1.0
    total_memory_gb = prob_matrix_gb + positions_gb + overhead_gb
    
    # Time estimation (in seconds per cluster)
    # Rough estimate: ~0.5 seconds per million pairs on CPU
    base_time_per_million = 0.5
    time_per_cluster = (n_pairs / 1e6) * base_time_per_million / n_workers
    total_time_hours = (time_per_cluster * n_clusters) / 3600
    
    return {
        "n_particles": n_particles,
        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "prob_matrix_gb": round(prob_matrix_gb, 2),
        "total_memory_gb": round(total_memory_gb, 2),
        "time_per_cluster_sec": round(time_per_cluster, 1),
        "estimated_total_hours": round(total_time_hours, 2),
    }
