"""
Unit tests for core mathematical functions in dna_linker.

These tests verify the correctness of the core scientific computations:
- Bending energy calculations
- Angle calculations
- Distance calculations
- Probability calculations
"""

import numpy as np
import pytest
import pandas as pd  # For DataFrame creation in tests
from scipy.spatial.transform import Rotation

# Import from dna_linker package
from dna_linker import config
from dna_linker.dna_linkers import (
    bending_energy,
    prob_bending_energy,
    prob_linker_length,
    calculate_angle,
    calculate_distance,
    calculate_probabilities,
    calculate_probabilities_all_connexions,
    calculate_probabilities_all_connexions_parallel,
    calculate_linker_length_connected,
    calculate_linker_length_connected_parallel,
    JOBLIB_AVAILABLE,
)


class TestBendingEnergy:
    """Tests for bending energy calculations."""
    
    def test_bending_energy_zero_angle(self):
        """Energy should be zero when angle is zero."""
        L = 100.0  # linker length in voxels
        lp = config.lp  # persistence length
        theta = 0.0
        energy = bending_energy(L=L, lp=lp, theta=theta, kB=config.kB, T=config.T)
        assert np.isclose(energy, 0.0, atol=1e-15)
    
    def test_bending_energy_proportional_to_theta_squared(self):
        """Energy should scale with theta^2."""
        L = 100.0
        lp = config.lp
        theta = 0.1
        energy = bending_energy(L=L, lp=lp, theta=theta, kB=config.kB, T=config.T)
        theta2 = 0.2
        energy2 = bending_energy(L=L, lp=lp, theta=theta2, kB=config.kB, T=config.T)
        # 0.2^2 = 4 * 0.1^2, so energy2 should be 4x energy
        assert np.isclose(energy2 / energy, 4.0, rtol=1e-10)
    
    def test_bending_energy_inversely_proportional_to_L(self):
        """Energy should scale inversely with linker length."""
        theta = 0.1
        L1 = 100.0
        L2 = 200.0
        energy1 = bending_energy(L=L1, lp=config.lp, theta=theta, kB=config.kB, T=config.T)
        energy2 = bending_energy(L=L2, lp=config.lp, theta=theta, kB=config.kB, T=config.T)
        # Energy at 200nm should be half of energy at 100nm
        assert np.isclose(energy1 / energy2, 2.0, rtol=1e-10)


class TestProbBendingEnergy:
    """Tests for bending energy probability calculations."""
    
    def test_probability_one_at_zero_angle(self):
        """Probability should be 1 when theta is 0."""
        theta = 0.0
        prob = prob_bending_energy(L=100.0, theta=theta, lp=config.lp)
        assert np.isclose(prob, 1.0, atol=1e-15)
    
    def test_probability_decreases_with_angle(self):
        """Probability should decrease as angle increases."""
        theta1 = 0.0
        theta2 = 0.5
        prob1 = prob_bending_energy(L=100.0, theta=theta1, lp=config.lp)
        prob2 = prob_bending_energy(L=100.0, theta=theta2, lp=config.lp)
        assert prob1 > prob2
    
    def test_probability_increases_with_length(self):
        """Probability should INCREASE with length (less bending penalty for longer linkers).
        
        This is because the bending energy term has L in the denominator:
        E ~ (lp/L) * theta^2, so longer linkers = less bending energy = higher probability.
        """
        L1 = 50.0
        L2 = 150.0
        theta = 0.2
        prob1 = prob_bending_energy(L=L1, theta=theta, lp=config.lp)
        prob2 = prob_bending_energy(L=L2, theta=theta, lp=config.lp)
        # Longer linker = less bending energy = higher probability
        assert prob1 < prob2, "Longer linkers should have higher probability"
    
    def test_probability_symmetric_around_zero(self):
        """Probability should be the same for +theta and -theta."""
        theta = 0.3
        prob_pos = prob_bending_energy(L=100.0, theta=theta, lp=config.lp)
        prob_neg = prob_bending_energy(L=100.0, theta=-theta, lp=config.lp)
        assert np.isclose(prob_pos, prob_neg, rtol=1e-15)


class TestProbLinkerLength:
    """Tests for linker length probability calculations."""
    
    def test_length_probability_one_at_zero(self):
        """Probability should be 1 when length is 0."""
        prob = prob_linker_length(L=0.0, lo=config.lo)
        assert np.isclose(prob, 1.0, atol=1e-15)
    
    def test_length_probability_exponential_decay(self):
        """Probability should follow exponential decay."""
        lo = config.lo
        L = lo
        prob = prob_linker_length(L=L, lo=lo)
        assert np.isclose(prob, np.exp(-1.0), rtol=1e-10)
    
    def test_length_probability_integration(self):
        """Probability should decrease monotonically."""
        L_values = np.array([0.0, 10.0, 50.0, 100.0, 200.0])
        probs = [prob_linker_length(L=L, lo=config.lo) for L in L_values]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], "Probability should decrease monotonically"


class TestCalculateAngle:
    """Tests for angle calculation between vectors."""
    
    def test_angle_zero_for_identical_vectors(self):
        """Angle between identical vectors should be 0."""
        v = np.array([1.0, 0.0, 0.0])
        angle = calculate_angle(v, v)
        assert np.isclose(angle, 0.0, atol=1e-15)
    
    def test_angle_pi_for_opposite_vectors(self):
        """Angle between opposite vectors should be pi."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        angle = calculate_angle(v1, v2)
        assert np.isclose(angle, np.pi, rtol=1e-10)
    
    def test_angle_right_angle(self):
        """Angle between perpendicular vectors should be pi/2."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        angle = calculate_angle(v1, v2)
        assert np.isclose(angle, np.pi / 2, rtol=1e-10)
    
    def test_angle_symmetric(self):
        """Angle should be symmetric: angle(v1, v2) = angle(v2, v1)."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        angle1 = calculate_angle(v1, v2)
        angle2 = calculate_angle(v2, v1)
        assert np.isclose(angle1, angle2, rtol=1e-15)
    
    def test_angle_preserves_norm(self):
        """Angle should be independent of vector magnitudes."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        v2_scaled = v2 * 10.0
        angle1 = calculate_angle(v1, v2)
        angle2 = calculate_angle(v1, v2_scaled)
        assert np.isclose(angle1, angle2, rtol=1e-15)
    
    def test_angle_random_vectors(self):
        """Test with random vectors."""
        np.random.seed(42)
        for _ in range(10):
            v1 = np.random.randn(3)
            v2 = np.random.randn(3)
            angle = calculate_angle(v1, v2)
            # Angle should be between 0 and pi
            assert 0 <= angle <= np.pi


class TestCalculateDistance:
    """Tests for Euclidean distance calculations."""
    
    def test_distance_zero_for_same_point(self):
        """Distance should be zero for identical points."""
        v = np.array([1.0, 2.0, 3.0])
        dist = calculate_distance(v, v)
        assert np.isclose(dist, 0.0, atol=1e-15)
    
    def test_distance_correct_for_simple_case(self):
        """Distance should be correct for simple case."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([3.0, 4.0, 0.0])
        dist = calculate_distance(v1, v2)
        assert np.isclose(dist, 5.0, rtol=1e-10)
    
    def test_distance_symmetric(self):
        """Distance should be symmetric."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        dist1 = calculate_distance(v1, v2)
        dist2 = calculate_distance(v2, v1)
        assert np.isclose(dist1, dist2, rtol=1e-15)
    
    def test_distance_triangle_inequality(self):
        """Distance should satisfy triangle inequality."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([1.0, 1.0, 0.0])
        d12 = calculate_distance(v1, v2)
        d23 = calculate_distance(v2, v3)
        d13 = calculate_distance(v1, v3)
        assert d13 <= d12 + d23 + 1e-10


class TestCalculateProbabilities:
    """Integration tests for full probability calculation."""
    
    def test_probability_physical_bounds(self):
        """Probability should be between 0 and 1."""
        np.random.seed(42)
        pos_selected = np.array([0.0, 0.0, 0.0])
        vector_selected = np.array([1.0, 0.0, 0.0])
        pos_current = np.array([10.0, 0.0, 0.0])
        vector_current = np.array([0.0, 1.0, 0.0])
        
        prob = calculate_probabilities(
            pos_selected=pos_selected,
            vector_selected=vector_selected,
            pos_current=pos_current,
            vector_current=vector_current,
            lo=config.lo
        )
        
        assert 0 <= prob <= 1
    
    def test_probability_high_for_aligned_vectors(self):
        """Probability should be relatively high for well-aligned particles.
        
        For particles to be "aligned", their vectors should point TOWARD each other
        along the connecting line.
        """
        # Particle at origin, pointing in +x direction
        pos_selected = np.array([0.0, 0.0, 0.0])
        vector_selected = np.array([1.0, 0.0, 0.0])  # Points +x
        
        # Particle at x=50, pointing in -x direction (toward first particle)
        pos_current = np.array([50.0, 0.0, 0.0])
        vector_current = np.array([-1.0, 0.0, 0.0])  # Points -x (toward first)
        
        prob = calculate_probabilities(
            pos_selected=pos_selected,
            vector_selected=vector_selected,
            pos_current=pos_current,
            vector_current=vector_current,
            lo=200.0  # Use longer expected length
        )
        
        # Aligned particles should have high probability
        assert prob > 0.1, f"Expected probability > 0.1, got {prob}"
    
    def test_probability_low_for_perpendicular_vectors(self):
        """Probability should be lower for perpendicular (high bending) vectors."""
        pos_selected = np.array([0.0, 0.0, 0.0])
        vector_selected = np.array([1.0, 0.0, 0.0])
        # Place particle 100 units away, pointing perpendicular
        pos_current = np.array([100.0, 0.0, 0.0])
        vector_current = np.array([0.0, 1.0, 0.0])  # Perpendicular
        
        prob = calculate_probabilities(
            pos_selected=pos_selected,
            vector_selected=vector_selected,
            pos_current=pos_current,
            vector_current=vector_current,
            lo=config.lo
        )
        
        # Should be lower due to high bending angle
        assert prob < 0.5, f"Expected probability < 0.5 for perpendicular, got {prob}"
    
    def test_probability_same_particle(self):
        """Same-particle probability is not a valid use case (callers filter this).
        
        When positions are identical, the connecting vector has zero norm,
        causing division by zero. Callers should exclude same-particle pairs.
        """
        pytest.skip("Same-particle probability is not a valid use case - callers filter these pairs")


@pytest.mark.skipif(not JOBLIB_AVAILABLE, reason="joblib is required for parallel tests")
class TestParallelProbabilityCalculation:
    """Tests for parallel probability calculation functions."""
    
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
    
    def test_parallel_vs_sequential_small(self):
        """Test that parallel and sequential versions produce identical results (N=10)."""
        np.random.seed(42)
        num = 10
        
        # Create synthetic test data
        motl = self._create_synthetic_motl(num, seed=42)
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
        
        # Run sequential version
        probs_seq = calculate_probabilities_all_connexions(
            motl=motl,
            motl_exit=motl_exit,
            motl_exit2=motl_exit2,
            motl_entry=motl_entry,
            motl_entry2=motl_entry2,
            lo=config.lo
        )
        
        # Run parallel version
        probs_par = calculate_probabilities_all_connexions_parallel(
            motl=motl,
            motl_exit=motl_exit,
            motl_exit2=motl_exit2,
            motl_entry=motl_entry,
            motl_entry2=motl_entry2,
            lo=config.lo,
            n_jobs=-1
        )
        
        # Compare results (use tolerance for floating point)
        np.testing.assert_allclose(probs_seq, probs_par, rtol=1e-10, atol=1e-15)
    
    def test_parallel_vs_sequential_medium(self):
        """Test that parallel and sequential versions produce identical results (N=20)."""
        np.random.seed(42)
        num = 20
        
        # Create synthetic test data
        motl = self._create_synthetic_motl(num, seed=42)
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
        
        # Run sequential version
        probs_seq = calculate_probabilities_all_connexions(
            motl=motl,
            motl_exit=motl_exit,
            motl_exit2=motl_exit2,
            motl_entry=motl_entry,
            motl_entry2=motl_entry2,
            lo=config.lo
        )
        
        # Run parallel version
        probs_par = calculate_probabilities_all_connexions_parallel(
            motl=motl,
            motl_exit=motl_exit,
            motl_exit2=motl_exit2,
            motl_entry=motl_entry,
            motl_entry2=motl_entry2,
            lo=config.lo,
            n_jobs=-1
        )
        
        # Compare results
        np.testing.assert_allclose(probs_seq, probs_par, rtol=1e-10, atol=1e-15)
    
    def test_parallel_probability_bounds(self):
        """Test that parallel version produces probabilities in valid range [0, 1]."""
        np.random.seed(42)
        num = 15
        
        # Create synthetic test data
        motl = self._create_synthetic_motl(num, seed=42)
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
        
        # Run parallel version
        probs_par = calculate_probabilities_all_connexions_parallel(
            motl=motl,
            motl_exit=motl_exit,
            motl_exit2=motl_exit2,
            motl_entry=motl_entry,
            motl_entry2=motl_entry2,
            lo=config.lo,
            n_jobs=-1
        )
        
        # All probabilities should be in [0, 1]
        assert np.all(probs_par >= 0.0), "Some probabilities are negative"
        assert np.all(probs_par <= 1.0), "Some probabilities are greater than 1"
        
        # Diagonal should be 0 (same particle)
        assert np.all(probs_par[np.eye(num, dtype=bool)] == 0.0), "Diagonal should be zero"
    
    def test_parallel_with_specific_n_jobs(self):
        """Test parallel version with specific number of jobs."""
        np.random.seed(42)
        num = 10
        
        # Create synthetic test data
        motl = self._create_synthetic_motl(num, seed=42)
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
        
        # Run parallel version with n_jobs=2
        probs_par = calculate_probabilities_all_connexions_parallel(
            motl=motl,
            motl_exit=motl_exit,
            motl_exit2=motl_exit2,
            motl_entry=motl_entry,
            motl_entry2=motl_entry2,
            lo=config.lo,
            n_jobs=2
        )
        
        # Compare with sequential version
        probs_seq = calculate_probabilities_all_connexions(
            motl=motl,
            motl_exit=motl_exit,
            motl_exit2=motl_exit2,
            motl_entry=motl_entry,
            motl_entry2=motl_entry2,
            lo=config.lo
        )
        
        np.testing.assert_allclose(probs_seq, probs_par, rtol=1e-10, atol=1e-15)


@pytest.mark.skipif(not JOBLIB_AVAILABLE, reason="joblib is required for parallel tests")
class TestParallelLinkerLength:
    """Tests for parallel linker length calculation functions."""
    
    def _create_synthetic_motl(self, num_particles, seed=42):
        """Create synthetic motl DataFrame with correct format."""
        from cryocat import cryomotl
        
        np.random.seed(seed)
        num = num_particles
        
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
    
    def test_linker_length_parallel_vs_sequential(self):
        """Test that parallel and sequential linker length calculations match."""
        np.random.seed(42)
        num = 10
        
        # Create synthetic motl data
        motl_exit = self._create_synthetic_motl(num, seed=42)
        motl_entry = self._create_synthetic_motl(num, seed=43)
        
        # Create synthetic connections dictionary - initialize all keys first
        connections = {i: [] for i in range(num)}
        for i in range(num):
            for j in range(i + 1, num):
                prob = np.random.rand()
                case = np.random.randint(0, 4)
                connections[i].append((j, prob, case))
                connections[j].append((i, prob, case))
        
        # Run sequential version
        lengths_seq = calculate_linker_length_connected(
            connections=connections,
            motl_exit=motl_exit,
            motl_entry=motl_entry,
            p_min=0.0  # Include all for testing
        )
        
        # Run parallel version
        lengths_par = calculate_linker_length_connected_parallel(
            connections=connections,
            motl_exit=motl_exit,
            motl_entry=motl_entry,
            p_min=0.0,
            n_jobs=-1
        )
        
        # Sort both arrays for comparison (order may differ)
        lengths_seq_sorted = np.sort(lengths_seq)
        lengths_par_sorted = np.sort(lengths_par)
        
        # Compare results
        np.testing.assert_allclose(lengths_seq_sorted, lengths_par_sorted, rtol=1e-10, atol=1e-15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

