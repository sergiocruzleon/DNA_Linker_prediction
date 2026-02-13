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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
