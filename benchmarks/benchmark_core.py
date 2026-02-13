"""
Benchmark harness for dna_linker.

This module provides benchmarking for:
1. Core mathematical functions
2. Probability calculations (the main bottleneck)
3. End-to-end pipeline performance

Usage:
    python benchmarks/benchmark_core.py
    pytest benchmarks/benchmark_core.py --benchmark-only
"""

import time
import cProfile
import pstats
import io
import numpy as np
import pandas as pd
from pathlib import Path
from functools import wraps

from dna_linker import config
from dna_linker.dna_linkers import (
    calculate_angle,
    calculate_distance,
    calculate_probabilities,
    bending_energy,
    prob_bending_energy,
    prob_linker_length,
    calculate_probabilities_all_connexions,
)


def time_function(func):
    """Decorator to time a function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper


class BenchmarkCoreFunctions:
    """Benchmarks for core mathematical functions."""
    
    def __init__(self):
        self.results = {}
        self.iterations = 10000
    
    def benchmark_calculate_angle(self):
        """Benchmark angle calculation."""
        np.random.seed(42)
        vectors = [np.random.randn(3) for _ in range(100)]
        
        @time_function
        def run():
            for v1 in vectors:
                for v2 in vectors:
                    calculate_angle(v1, v2)
        
        result, elapsed = run()
        avg_time = elapsed / (len(vectors) ** 2)
        self.results["calculate_angle"] = {
            "total_time": elapsed,
            "avg_time": avg_time,
            "calls_per_sec": 1 / avg_time if avg_time > 0 else 0,
        }
        return self.results["calculate_angle"]
    
    def benchmark_calculate_distance(self):
        """Benchmark distance calculation."""
        np.random.seed(42)
        vectors = [np.random.randn(3) for _ in range(100)]
        
        @time_function
        def run():
            for v1 in vectors:
                for v2 in vectors:
                    calculate_distance(v1, v2)
        
        result, elapsed = run()
        avg_time = elapsed / (len(vectors) ** 2)
        self.results["calculate_distance"] = {
            "total_time": elapsed,
            "avg_time": avg_time,
            "calls_per_sec": 1 / avg_time if avg_time > 0 else 0,
        }
        return self.results["calculate_distance"]
    
    def benchmark_calculate_probabilities(self):
        """Benchmark single probability calculation."""
        np.random.seed(42)
        pos_selected = np.array([0.0, 0.0, 0.0])
        vector_selected = np.array([1.0, 0.0, 0.0])
        
        @time_function
        def run():
            for _ in range(self.iterations):
                pos_current = np.random.randn(3) * 100
                vector_current = np.random.randn(3)
                vector_current = vector_current / np.linalg.norm(vector_current)
                calculate_probabilities(
                    pos_selected=pos_selected,
                    vector_selected=vector_selected,
                    pos_current=pos_current,
                    vector_current=vector_current,
                    lo=config.lo
                )
        
        result, elapsed = run()
        avg_time = elapsed / self.iterations
        self.results["calculate_probabilities"] = {
            "total_time": elapsed,
            "avg_time": avg_time,
            "calls_per_sec": self.iterations / elapsed,
        }
        return self.results["calculate_probabilities"]


class BenchmarkProbabilityMatrix:
    """Benchmark probability matrix calculation (the main bottleneck)."""
    
    def __init__(self):
        self.results = {}
    
    def _create_synthetic_motl(self, num_particles, seed=42):
        """Create synthetic motl DataFrame with correct format."""
        from cryocat import cryomotl
        
        np.random.seed(seed)
        num = num_particles
        
        # Create synthetic motl dataframes with correct cryomotl format
        df = pd.DataFrame({
            'score': np.ones(num),
            'geom1': np.zeros(num),
            'geom2': np.zeros(num),
            'subtomo_id': list(range(num)),
            'tomo_id': [0] * num,
            'object_id': list(range(num)),
            'subtomo_mean': np.zeros(num),
            'x': np.random.randn(num) * 100,
            'y': np.random.randn(num) * 100,
            'z': np.random.randn(num) * 100,
            'shift_x': np.zeros(num),
            'shift_y': np.zeros(num),
            'shift_z': np.zeros(num),
            'geom3': np.zeros(num),
            'geom4': np.zeros(num),
            'geom5': np.zeros(num),
            'phi': np.zeros(num),
            'psi': np.zeros(num),
            'theta': np.zeros(num),
            'class': np.zeros(num),
        })
        
        return cryomotl.Motl(df)
    
    def benchmark_with_profiler(self, num_particles=50):
        """Run probability calculation with profiling."""
        np.random.seed(42)
        num = num_particles
        
        # Create synthetic test data
        motl = self._create_synthetic_motl(num, seed=42)
        
        # Create entry/exit variations
        motl_exit = self._create_synthetic_motl(num, seed=43)
        motl_entry = self._create_synthetic_motl(num, seed=44)
        
        # Create exit2/entry2 with small offsets
        df_exit = motl_exit.df.copy()
        df_exit2 = df_exit.copy()
        df_entry = motl_entry.df.copy()
        df_entry2 = df_entry.copy()
        
        for i in range(num):
            offset = np.random.randn(3) * 5
            df_exit2.loc[i, ['x', 'y', 'z']] = df_exit.loc[i, ['x', 'y', 'z']] + offset
            df_entry2.loc[i, ['x', 'y', 'z']] = df_entry.loc[i, ['x', 'y', 'z']] + offset
        
        from cryocat import cryomotl
        motl_exit2 = cryomotl.Motl(df_exit2)
        motl_entry2 = cryomotl.Motl(df_entry2)
        
        # Profile the calculation
        profiler = cProfile.Profile()
        profiler.enable()
        
        start = time.perf_counter()
        probs = calculate_probabilities_all_connexions(
            motl=motl,
            motl_exit=motl_exit,
            motl_exit2=motl_exit2,
            motl_entry=motl_entry,
            motl_entry2=motl_entry2,
            lo=config.lo
        )
        elapsed = time.perf_counter() - start
        
        profiler.disable()
        
        # Parse profiling results
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        self.results[f"prob_matrix_{num_particles}"] = {
            "num_particles": num_particles,
            "total_time": elapsed,
            "matrix_shape": probs.shape,
            "profiling": s.getvalue(),
        }
        
        return self.results[f"prob_matrix_{num_particles}"]
    
    def benchmark_scaling(self, particle_counts=[10, 20, 30, 50]):
        """Benchmark how runtime scales with particle count."""
        scaling_results = {}
        
        for n in particle_counts:
            print(f"Benchmarking with {n} particles...")
            result = self.benchmark_with_profiler(num_particles=n)
            scaling_results[n] = {
                "time": result["total_time"],
            }
            print(f"  Time: {result['total_time']:.4f}s")
        
        # Calculate scaling factor
        if len(scaling_results) >= 2:
            times = [scaling_results[n]["time"] for n in particle_counts]
            # Expected O(N^2) scaling - times should roughly scale with N^2
            for i, n in enumerate(particle_counts):
                expected = times[0] * (n / particle_counts[0]) ** 2
                actual = times[i]
                scaling_results[n]["expected_o_n2"] = expected
                scaling_results[n]["speedup_factor"] = expected / actual if actual > 0 else 0
        
        self.results["scaling"] = scaling_results
        return scaling_results


class BenchmarkEndToEnd:
    """End-to-end pipeline benchmarks."""
    
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def run_benchmarks(self):
        """Run all benchmarks and save results."""
        core = BenchmarkCoreFunctions()
        matrix = BenchmarkProbabilityMatrix()
        
        print("=" * 60)
        print("DNA_LINKER BENCHMARK SUITE")
        print("=" * 60)
        
        # Core function benchmarks
        print("\n[1] Core Function Benchmarks")
        print("-" * 40)
        
        print("  calculate_angle...")
        core.benchmark_calculate_angle()
        print(f"    Avg time: {core.results['calculate_angle']['avg_time']*1e6:.2f} µs")
        
        print("  calculate_distance...")
        core.benchmark_calculate_distance()
        print(f"    Avg time: {core.results['calculate_distance']['avg_time']*1e6:.2f} µs")
        
        print("  calculate_probabilities...")
        core.benchmark_calculate_probabilities()
        print(f"    Avg time: {core.results['calculate_probabilities']['avg_time']*1e6:.2f} µs")
        
        # Probability matrix benchmarks
        print("\n[2] Probability Matrix Benchmarks")
        print("-" * 40)
        
        print("  Scaling test (10, 20, 30, 50 particles)...")
        scaling = matrix.benchmark_scaling([10, 20, 30, 50])
        
        print("\n  Scaling Results:")
        print("  " + "-" * 36)
        print(f"  {'Particles':<12} {'Time (s)':<12} {'Expected O(N²)':<15} {'Ratio':<10}")
        print("  " + "-" * 36)
        for n, data in scaling.items():
            expected = data.get("expected_o_n2", 0)
            ratio = data.get("speedup_factor", 0)
            print(f"  {n:<12} {data['time']:<12.4f} {expected:<15.4f} {ratio:<10.2f}x")
        
        # Save results
        import json
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "core": core.results,
                "matrix": matrix.results,
            }, f, indent=2)
        
        print(f"\n[✓] Results saved to {results_file}")
        
        return core.results, matrix.results


def profile_probability_calculation(num_particles=50):
    """Profile the probability calculation function."""
    from cryocat import cryomotl
    
    np.random.seed(42)
    num = num_particles
    
    # Create synthetic data
    df = pd.DataFrame({
        'score': np.ones(num),
        'geom1': np.zeros(num),
        'geom2': np.zeros(num),
        'subtomo_id': list(range(num)),
        'tomo_id': [0] * num,
        'object_id': list(range(num)),
        'subtomo_mean': np.zeros(num),
        'x': np.random.randn(num) * 100,
        'y': np.random.randn(num) * 100,
        'z': np.random.randn(num) * 100,
        'shift_x': np.zeros(num),
        'shift_y': np.zeros(num),
        'shift_z': np.zeros(num),
        'geom3': np.zeros(num),
        'geom4': np.zeros(num),
        'geom5': np.zeros(num),
        'phi': np.zeros(num),
        'psi': np.zeros(num),
        'theta': np.zeros(num),
        'class': np.zeros(num),
    })
    
    motl = cryomotl.Motl(df)
    motl_exit = cryomotl.Motl(df.copy())
    motl_exit2 = cryomotl.Motl(df.copy())
    motl_entry = cryomotl.Motl(df.copy())
    motl_entry2 = cryomotl.Motl(df.copy())
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    probs = calculate_probabilities_all_connexions(
        motl=motl,
        motl_exit=motl_exit,
        motl_exit2=motl_exit2,
        motl_entry=motl_entry,
        motl_entry2=motl_entry2,
        lo=config.lo
    )
    
    profiler.disable()
    
    # Print stats
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30)
    
    print("=" * 60)
    print(f"PROFILING RESULTS: {num_particles} particles")
    print("=" * 60)
    print(s.getvalue())
    
    return probs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run dna_linker benchmarks")
    parser.add_argument("--profile", action="store_true", help="Run with profiling")
    parser.add_argument("--particles", type=int, default=50, help="Number of particles for profiling")
    parser.add_argument("--scaling", action="store_true", help="Run scaling test")
    args = parser.parse_args()
    
    if args.profile:
        profile_probability_calculation(num_particles=args.particles)
    elif args.scaling:
        bench = BenchmarkEndToEnd()
        bench.run_benchmarks()
    else:
        bench = BenchmarkEndToEnd()
        bench.run_benchmarks()
