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

from dna_linker import run_pipeline as run
from dna_linker import config


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
        default="./DNA_Linker_prediction/inputs",
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
        default="./DNA_Linker_prediction/outputs",
        help="Base output directory"
    )
    
    # Benchmarking
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Run in benchmark mode (single worker, verbose timing)"
    )
    
    return parser.parse_args()


def run_single_emd(
    emd_id: int,
    suffix: str,
    input_dir: str,
    output_base: str,
    workers: int = 1,
    skip_tracing: bool = False
):
    """Run pipeline for a single EMD dataset."""
    from cryocat import cryomotl
    
    print(f"\n{'='*60}")
    print(f"Processing EMD{emd_id} with suffix '{suffix}'")
    print(f"{'='*60}")
    
    # Configuration
    path_mask = input_dir
    entry = "Threshold_ref_entrymask_r2_resamp_righthand.mrc"
    exit = "Threshold_ref_exitmask_r2_resamp_righthand.mrc"
    origin_entry = "Threshold_ref_Origin_entrymask_r2_resamp_righthand.mrc"
    origin_exit = "Threshold_ref_Origin_exitmask_r2_resamp_righthand.mrc"
    
    tracing_distance = config.tracing_distance
    pixel_size = config.pixel_size
    bin_factor = config.bin
    max_distance = tracing_distance / (pixel_size * bin_factor)
    
    # Paths
    motl_name = f"motl_EMD{emd_id}_{suffix}.em"
    name_traced = f"EMD{emd_id}_tr{int(tracing_distance)}nm_{suffix}.em"
    path_output = f"{output_base}/EMD{emd_id}_{suffix}/"
    motl_trace_input = path_output + name_traced
    
    output_path_cluster = f"{output_base}/outputs_EMD{emd_id}_{suffix}/clusters_20nm/"
    output_path_linker = f"{output_base}/outputs_EMD{emd_id}_{suffix}/A_linkers_20nm/"
    output_path_dictionary = f"{output_base}/outputs_EMD{emd_id}_{suffix}/A_Connections_dictionary_20nm/"
    
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
            dnal_object=config.lo,
            max_processes=workers,
        )
    
    # Report results
    print("\n  Results:")
    for f in Path(output_path_cluster).glob("motl_tomo*.em"):
        motl = cryomotl.EmMotl(input_motl=str(f))
        print(f"    {f.name}: {len(motl.df)} particles")
    
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("DAN_LINKER PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Determine EMDs to process
    if args.all:
        emd_list = [2601, 13356, 13363, 13370, 13379, 38407]
    else:
        emd_list = args.emd
    
    # Run for each EMD
    start_time = datetime.now()
    success_count = 0
    
    for emd in emd_list:
        try:
            success = run_single_emd(
                emd_id=emd,
                suffix=args.suffix,
                input_dir=args.input_dir,
                output_base=args.output_base,
                workers=args.workers,
                skip_tracing=args.skip_tracing
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
