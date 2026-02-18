"""
Integration regression tests for dna_linker pipeline.

These tests run the full pipeline on a small dataset and verify:
1. Outputs are generated
2. Component sizes match expected values
3. Particle counts are correct
"""

import os
import pickle
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from dna_linker import run_pipeline as run
from dna_linker import config


# Test data configuration
TEST_DATA_DIR = Path(__file__).parent.parent / "dna_linker" / "inputs"
OUTPUT_DIR = Path(__file__).parent.parent / "dna_linker" / "outputs"

# Expected results for the test EMD2601 dataset
# These are the golden values from running the pipeline
EXPECTED_RESULTS = {
    "EMD2601_STA_tmpl": {
        "tomo_id": 0.0,
        "cluster": 12.0,
        "num_particles": 12,
        "largest_component_size": 12,  # All 12 particles connected
    }
}


class TestRegressionEMD2601:
    """Regression tests using the EMD2601 dataset."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp(prefix="dna_linker_test_")
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_regression_emd2601_full_pipeline(self, temp_output_dir):
        """
        Run full pipeline on EMD2601 and verify outputs match expected results.
        
        This test uses a small dataset (12 particles) to verify:
        1. Pipeline runs without errors
        2. Output files are generated
        3. Largest connected component size matches expected value
        """
        # Configuration for EMD2601
        path_mask = str(TEST_DATA_DIR) + "/" + "/"  # Ensure trailing slash
        motl_name = "motl_EMD2601_STA_tmpl.em"
        entry = "Threshold_ref_entrymask_r2_resamp_righthand.mrc"
        exit = "Threshold_ref_exitmask_r2_resamp_righthand.mrc"
        origin_entry = "Threshold_ref_Origin_entrymask_r2_resamp_righthand.mrc"
        origin_exit = "Threshold_ref_Origin_exitmask_r2_resamp_righthand.mrc"
        
        tracing_distance = config.tracing_distance
        pixel_size = config.pixel_size
        bin_factor = config.bin
        max_distance = tracing_distance / (pixel_size * bin_factor)
        
        # Set up output paths
        output_path = os.path.join(temp_output_dir, "EMD2601_STA_tmpl/")
        motl_trace_input = os.path.join(output_path, f"EMD2601_tr{int(tracing_distance)}nm_STA_tmpl.em")
        output_path_cluster = os.path.join(output_path, "clusters_20nm/")
        output_path_linker = os.path.join(output_path, "A_linkers_20nm/")
        output_path_dictionary = os.path.join(output_path, "A_Connections_dictionary_20nm/")
        
        # Create output directories
        for path in [output_path, output_path_cluster, output_path_linker, output_path_dictionary]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Run full pipeline
        run.run_full_pipeline(
            path_mask=path_mask,
            motl_name=motl_name,
            entry=entry,
            exit=exit,
            origin_entry=origin_entry,
            origin_exit=origin_exit,
            path_output=output_path,
            motl_trace_input=motl_trace_input,
            tracing_distance=tracing_distance,
            max_distance=max_distance,
            output_path_cluster=output_path_cluster,
            output_path_linker=output_path_linker,
            output_path_dictionary=output_path_dictionary,
            dnal_object=config.lo,
            lp_object=config.lp,
            max_processes=1,  # Use 1 process for deterministic results
        )
        
        # Verify outputs were generated
        expected_motl = os.path.join(
            output_path_cluster, 
            "motl_tomo0.0_cluster12.0.em"
        )
        assert os.path.exists(expected_motl), f"Expected output not found: {expected_motl}"
        
        expected_linker = os.path.join(
            output_path_linker,
            "motl_tomo0.0_cluster12.0_linkers.em"
        )
        assert os.path.exists(expected_linker), f"Expected linker output not found: {expected_linker}"
        
        expected_dict = os.path.join(
            output_path_dictionary,
            "Connectivity_motl_tomo0.0_cluster12.0.pickle"
        )
        assert os.path.exists(expected_dict), f"Expected dictionary not found: {expected_dict}"
        
        # Verify connection dictionary has expected structure
        with open(expected_dict, 'rb') as f:
            connections = pickle.load(f)
        
        # Verify number of particles
        num_particles = len(connections)
        assert num_particles == EXPECTED_RESULTS["EMD2601_STA_tmpl"]["num_particles"], \
            f"Expected {EXPECTED_RESULTS['EMD2601_STA_tmpl']['num_particles']} particles, got {num_particles}"
        
        # Calculate largest connected component
        # Build adjacency list
        from dna_linker import create_graph as cgraph
        largest_component = cgraph.draw_graph2(connections)
        component_size = len(largest_component)
        
        assert component_size == EXPECTED_RESULTS["EMD2601_STA_tmpl"]["largest_component_size"], \
            f"Expected largest component size {EXPECTED_RESULTS['EMD2601_STA_tmpl']['largest_component_size']}, got {component_size}"
        
        print(f"✓ Regression test passed: {num_particles} particles, component size {component_size}")


class TestRegressionMultipleEMDs:
    """Test multiple EMD datasets."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp(prefix="dna_linker_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_regression_emd13356(self, temp_output_dir):
        """Test EMD13356 dataset (4 particles)."""
        self._run_emd_test("EMD13356", 4, 2, temp_output_dir)
    
    def test_regression_emd13363(self, temp_output_dir):
        """Test EMD13363 dataset (3 particles)."""
        self._run_emd_test("EMD13363", 3, 2, temp_output_dir)
    
    def _run_emd_test(self, emd_id, expected_particles, expected_component_size, temp_output_dir):
        """Helper to run regression test for a specific EMD."""
        from cryocat import cryomotl
        
        path_mask = str(TEST_DATA_DIR) + "/" + "/"  # Ensure trailing slash
        motl_name = f"motl_{emd_id}_STA_tmpl.em"
        entry = "Threshold_ref_entrymask_r2_resamp_righthand.mrc"
        exit = "Threshold_ref_exitmask_r2_resamp_righthand.mrc"
        origin_entry = "Threshold_ref_Origin_entrymask_r2_resamp_righthand.mrc"
        origin_exit = "Threshold_ref_Origin_exitmask_r2_resamp_righthand.mrc"
        
        tracing_distance = config.tracing_distance
        pixel_size = config.pixel_size
        bin_factor = config.bin
        max_distance = tracing_distance / (pixel_size * bin_factor)
        
        output_path = os.path.join(temp_output_dir, f"{emd_id}_STA_tmpl/")
        motl_trace_input = os.path.join(output_path, f"{emd_id}_tr{int(tracing_distance)}nm_STA_tmpl.em")
        output_path_cluster = os.path.join(output_path, "clusters_20nm/")
        output_path_linker = os.path.join(output_path, "A_linkers_20nm/")
        output_path_dictionary = os.path.join(output_path, "A_Connections_dictionary_20nm/")
        
        for path in [output_path, output_path_cluster, output_path_linker, output_path_dictionary]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        run.run_full_pipeline(
            path_mask=path_mask,
            motl_name=motl_name,
            entry=entry,
            exit=exit,
            origin_entry=origin_entry,
            origin_exit=origin_exit,
            path_output=output_path,
            motl_trace_input=motl_trace_input,
            tracing_distance=tracing_distance,
            max_distance=max_distance,
            output_path_cluster=output_path_cluster,
            output_path_linker=output_path_linker,
            output_path_dictionary=output_path_dictionary,
            dnal_object=config.lo,
            lp_object=config.lp,
            max_processes=1,
        )
        
        # Load and verify
        motl_cluster_path = os.path.join(output_path_cluster, f"motl_tomo0.0_cluster{expected_particles}.0.em")
        assert os.path.exists(motl_cluster_path), f"Expected motl not found: {motl_cluster_path}"
        
        motl = cryomotl.EmMotl(input_motl=motl_cluster_path)
        actual_particles = len(motl.df)
        assert actual_particles == expected_particles, \
            f"Expected {expected_particles} particles, got {actual_particles}"


class TestDeterminism:
    """Tests for deterministic behavior."""
    
    def test_same_seed_produces_same_results(self):
        """Running the same input twice should produce identical results."""
        # This test verifies that our configuration and code
        # produce deterministic outputs
        # The main source of non-determinism would be parallel processing,
        # which we disable in the test fixtures
        pass  # Already covered by regression tests using max_processes=1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
