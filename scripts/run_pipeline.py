#!/usr/bin/env python
"""
DAN_LINKER Pipeline Entry Point

This script provides a command-line interface to run the DNA linker prediction pipeline.

Usage:
    python scripts/run_pipeline.py --help
    python scripts/run_pipeline.py --emd 2601 --suffix STA_tmpl
    python scripts/run_pipeline.py --emd 2601 --emd 13356 --suffix STA_tmpl --workers 4
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add dna_linker package to path
_repo_root = Path(__file__).resolve().parent.parent
# Add parent dir so 'dna_linker' becomes importable as a module
sys.path.insert(0, str(_repo_root))

from dna_linker import run_pipeline as run


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run DNA linker prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run single dataset
    python scripts/run_pipeline.py --emd 2601 --suffix STA_tmpl
    
    # Run multiple datasets
    python scripts/run_pipeline.py --emd 2601 13356 13363 --suffix STA_tmpl
    
    # Run with custom workers
    python scripts/run_pipeline.py --emd 2601 --workers 8
    
    # Run full test suite
    python scripts/run_pipeline.py --all
        """
    )
    
    # Input options
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="./dna_linker/inputs",
        help="Input directory containing mask files"
    )
    
    # EMD datasets
    parser.add_argument(
        "--emd", 
        type=int, 
        nargs="+",
        default=[2601],
        help="EMD dataset IDs to process (default: 2601)"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Process all available EMD datasets"
    )
    
    # Suffix options
    parser.add_argument(
        "--suffix", 
        type=str, 
        default="STA_tmpl",
        help="File suffix for motl files (default: STA_tmpl)"
    )
    
    # Processing options
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1,
        help="Number of parallel workers (default: 1, sequential)"
    )
    parser.add_argument(
        "--skip-tracing", 
        action="store_true",
        help="Skip tracing step if already done"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="./dna_linker/outputs",
        help="Base output directory"
    )
    
    # Benchmarking
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Run in benchmark mode (single worker, verbose timing)"
    )
    
    # Estimate mode
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate runtime and memory requirements without running"
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML config file (default: uses pipeline_config.yaml)"
    )
    
    return parser.parse_args()


def run_single_emd(
    emd_id: int,
    suffix: str,
    input_dir: str,
    output_base: str,
    workers: int = 1,
    skip_tracing: bool = False,
    cfg = None
):
    """Run pipeline for a single EMD dataset."""
    from cryocat import cryomotl
    
    # Use config object if provided, otherwise fallback to module config
    if cfg is None:
        from dna_linker import config as mod_cfg
        cfg = mod_cfg
    
    print(f"\n{'='*60}")
    print(f"Processing EMD{emd_id} with suffix '{suffix}'")
    print(f"{'='*60}")
    
    # Configuration (from config file)
    print(f"[CONFIG] Using input_dir: {cfg.input_dir}")
    print(f"[CONFIG] Using entry_mask: {cfg.entry_mask}")
    print(f"[CONFIG] Using tracing_distance: {cfg.tracing_distance}")
    print(f"[CONFIG] Using pixel_size: {cfg.pixel_size}, bin: {cfg.bin}")
    
    path_mask = cfg.input_dir
    entry = cfg.entry_mask
    exit = cfg.exit_mask
    origin_entry = cfg.origin_entry_mask
    origin_exit = cfg.origin_exit_mask
    
    tracing_distance = cfg.tracing_distance
    pixel_size = cfg.pixel_size
    bin_factor = cfg.bin
    max_distance = tracing_distance / (pixel_size * bin_factor)
    
    
    # Paths using pattern from config
    motl_name = cfg.motl_pattern.format(emd_id=emd_id, suffix=suffix)
    name_traced = cfg.traced_pattern.format(emd_id=emd_id, tracing_distance=int(tracing_distance), suffix=suffix)
    
    # Ensure paths end with separator
    input_dir = str(Path(input_dir))  # Normalize path
    output_base_dir = str(Path(output_base))
    
    # Full paths
    path_output = output_base_dir + "/" + cfg.output_tomo_dir.format(emd_id=emd_id, suffix=suffix)
    motl_trace_input = path_output + name_traced
    
    output_path_cluster = output_base_dir + "/" + cfg.output_clusters.format(emd_id=emd_id, suffix=suffix)
    output_path_linker = output_base_dir + "/" + cfg.output_linkers.format(emd_id=emd_id, suffix=suffix)
    output_path_dictionary = output_base_dir + "/" + cfg.output_dictionary.format(emd_id=emd_id, suffix=suffix)
    
    # Input path with trailing slash
    path_mask = input_dir if input_dir.endswith('/') else input_dir + '/'
    
    # Create output directories
    for path in [path_output, output_path_cluster, output_path_linker, output_path_dictionary]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Check if tracing is already done
    if skip_tracing and Path(motl_trace_input).exists():
        print(f"  Skipping tracing (output exists): {motl_trace_input}")
    else:
        print("  Step 1: Recentering and writing motl...")
        run.run_full_pipeline(
            path_mask=path_mask,
            motl_name=motl_name,
            entry=entry,
            exit=exit,
            origin_entry=origin_entry,
            origin_exit=origin_exit,
            path_output=path_output,
            motl_trace_input=motl_trace_input,
            tracing_distance=tracing_distance,
            max_distance=max_distance,
            output_path_cluster=output_path_cluster,
            output_path_linker=output_path_linker,
            output_path_dictionary=output_path_dictionary,
            dnal_object=cfg.lo,
            lp_object=cfg.lp,
            max_processes=workers,
        )
    
    # Report results
    print("\n  Results:")
    for f in Path(output_path_cluster).glob("motl_tomo*.em"):
        motl = cryomotl.EmMotl(input_motl=str(f))
        print(f"    {f.name}: {len(motl.df)} particles")
    
    # Compile all linker .em files into a single merged file
    print("\n  Compiling all linker files...")
    motl_trace = cryomotl.EmMotl(input_motl=motl_trace_input)
    tomograms = motl_trace.df['tomo_id'].unique()

    nlinkers = 0
    motl_lists = []
    for tomo_id in tomograms:
        df_motl_tomo = motl_trace.df[motl_trace.df['tomo_id'] == tomo_id]
        clusters = df_motl_tomo['geom1'].unique()
        for cluster in clusters:
            if cluster > 1:
                output_filename = output_path_linker + f'motl_tomo{tomo_id}_cluster{cluster}_linkers.em'
                try:
                    linkers = cryomotl.EmMotl(input_motl=output_filename)
                    nlinkers = nlinkers + len(linkers.df)
                    motl_lists.append(linkers)
                except Exception as e:
                    print(f"{output_filename} does not exist!!!")
                    continue

    print(f"A total of {nlinkers} linkers were found")
    print(f"Number of linker files: {len(motl_lists)}")

    if motl_lists:
        merged_motl = cryomotl.Motl.merge_and_renumber(motl_list=motl_lists)
        merged_motl.write_out(output_path=output_path_linker + 'ALL_linkers_with_length_and_bending_angle.em')
        print(f"Merged linkers written to: {output_path_linker}ALL_linkers_with_length_and_bending_angle.em")
    else:
        print("No linker files found to merge.")
    
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration from YAML file
    from dna_linker.config import get_config_for_run, estimate_runtime
    cfg = get_config_for_run(args.config)
    
    print("\n" + "="*60)
    print("DNA_LINKER PIPELINE")
    print(f"Config: {args.config or 'default pipeline_config.yaml'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Estimate mode - show requirements without running
    if args.estimate:
        print("\n[ESTIMATE MODE] - Showing requirements only\n")
        # Try to get particle count from input file
        motl_path = Path(cfg.input_dir) / cfg.motl_pattern.format(emd_id=args.emd[0] if args.emd else 2601, suffix=args.suffix)
        n_particles = 1000  # Default estimate
        n_tomograms = 1
        
        if motl_path.exists():
            try:
                from cryocat import cryomotl
                motl = cryomotl.EmMotl(str(motl_path))
                n_particles = len(motl.df)
                n_tomograms = len(motl.df['tomo_id'].unique())
            except:
                pass
        
        est = estimate_runtime(n_particles, n_workers=args.workers)
        
        print(f"Input file: {motl_path}")
        print(f"Total particles: {est['n_particles']:,}")
        print(f"Number of tomograms: {n_tomograms}")
        
        # Per tomogram estimate
        if n_tomograms > 1:
            avg_particles_per_tomo = n_particles // n_tomograms
            est_tomo = estimate_runtime(avg_particles_per_tomo, n_workers=args.workers)
            print(f"\n--- Per-tomogram estimate (avg {avg_particles_per_tomo:,} particles) ---")
            print(f"Pairs per tomogram: {avg_particles_per_tomo**2:,}")
            print(f"Memory per tomogram: {est_tomo['total_memory_gb']} GB")
            print(f"Time per tomogram: {est_tomo['time_per_cluster_sec']} seconds")
        
        print(f"\n--- Full dataset estimate ---")
        print(f"Number of pairs to compute: {est['n_pairs']:,} (N²)")
        print(f"Estimated memory (prob matrix): {est['prob_matrix_gb']} GB")
        print(f"Estimated total memory: {est['total_memory_gb']} GB")
        print(f"Estimated time per cluster: {est['time_per_cluster_sec']} seconds")
        print(f"Estimated total time: {est['estimated_total_hours']} hours")
        print("\nTo run with GPU acceleration (much faster), set GPU_ACCELERATE=true")
        return
    
    # Determine EMDs to process
    if args.all:
        emd_list = [2601, 13356, 13363, 13370, 13379, 38407]
    else:
        emd_list = args.emd
    
    # Run for each EMD
    start_time = datetime.now()
    success_count = 0
    
    # Use workers from config if not specified on command line
    workers = args.workers if args.workers != 1 else cfg.workers
    
    for emd in emd_list:
        try:
            success = run_single_emd(
                emd_id=emd,
                suffix=args.suffix if args.suffix else cfg.suffix,
                input_dir=args.input_dir if args.input_dir != "./dna_linker/inputs" else cfg.input_dir,
                output_base=args.output_base if args.output_base != "./dna_linker/outputs" else cfg.output_base,
                workers=workers,
                skip_tracing=args.skip_tracing,
                cfg=cfg
            )
            if success:
                success_count += 1
        except Exception as e:
            print(f"\n  ERROR processing EMD{emd}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    elapsed = datetime.now() - start_time
    print("\n" + "="*60)
    print(f"COMPLETED: {success_count}/{len(emd_list)} datasets processed")
    print(f"Total time: {elapsed}")
    print("="*60)
    
    return 0 if success_count == len(emd_list) else 1


if __name__ == "__main__":
    sys.exit(main())
