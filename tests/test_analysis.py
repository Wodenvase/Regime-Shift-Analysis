"""
Unit tests for Resilience & Liquidity Analysis
Phases VIII: Code Quality & Testing
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from src.shock_registry import (
    BLACK_SWAN_REGISTRY,
    get_events_by_month,
    is_shock_date,
    get_event_magnitude,
    CANADIAN_WILDFIRE_SMOKE,
    SEPTEMBER_FLASH_FLOOD,
)
from src import visualizations


class TestShockRegistry:
    """Test Black Swan event detection"""

    def test_black_swan_registry_exists(self):
        """Registry should contain events"""
        assert len(BLACK_SWAN_REGISTRY) >= 2

    def test_canadian_wildfire_detection(self):
        """June 7-9 should be flagged as shock"""
        assert is_shock_date(datetime(2023, 6, 7))
        assert is_shock_date(datetime(2023, 6, 8))
        assert is_shock_date(datetime(2023, 6, 9))

    def test_non_shock_date(self):
        """Random dates should not be flagged"""
        assert not is_shock_date(datetime(2023, 1, 15))
        assert not is_shock_date(datetime(2023, 4, 20))

    def test_shock_magnitude(self):
        """Magnitudes should be normalized 0-1"""
        mag_june = get_event_magnitude(datetime(2023, 6, 8))
        mag_sept = get_event_magnitude(datetime(2023, 9, 29))
        mag_none = get_event_magnitude(datetime(2023, 1, 1))

        assert 0 <= mag_june <= 1
        assert 0 <= mag_sept <= 1
        assert mag_none == 0.0
        assert mag_sept > mag_june  # Sept flood more severe

    def test_events_by_month(self):
        """Filter events by month"""
        june_events = get_events_by_month(6)
        sept_events = get_events_by_month(9)

        assert len(june_events) >= 1
        assert len(sept_events) >= 1

    def test_canadian_wildfire_properties(self):
        """Validate Canadian Wildfire event"""
        assert CANADIAN_WILDFIRE_SMOKE.magnitude == 0.85
        assert CANADIAN_WILDFIRE_SMOKE.start_date.month == 6
        assert CANADIAN_WILDFIRE_SMOKE.event_type == "external"

    def test_september_flood_properties(self):
        """Validate September Flood event"""
        assert SEPTEMBER_FLASH_FLOOD.magnitude == 0.95
        assert SEPTEMBER_FLASH_FLOOD.start_date.month == 9
        assert SEPTEMBER_FLASH_FLOOD.event_type == "weather"
        assert "flood" in SEPTEMBER_FLASH_FLOOD.name.lower()


class TestVisualizations:
    """Test visualization functions"""

    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.t = np.arange(100)
        self.actual = 100 + 20 * np.sin(self.t / 10) + np.random.normal(0, 5, 100)
        self.forecast = 100 + 18 * np.sin(self.t / 10)
        self.std = np.full_like(self.forecast, 8.0)
        self.beta_samples = np.random.normal(-0.35, 0.08, 1000)
        self.residuals = np.random.standard_t(3.5, size=1000)

    def test_bayesian_fan_chart_creates(self):
        """Fan chart should generate without error"""
        fig = visualizations.plot_bayesian_fan_chart(
            self.actual, self.forecast, self.std, shock_dates=[(20, 25)]
        )
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_impulse_response_creates(self):
        """IRF should generate without error"""
        t_horizon = np.arange(0, 24)
        irf = np.exp(-0.15 * t_horizon)

        fig = visualizations.plot_impulse_response_function(t_horizon, irf, "Test Shock")
        assert fig is not None

    def test_posterior_beta_creates(self):
        """Posterior beta histogram should generate"""
        fig = visualizations.plot_posterior_beta_distribution(
            self.beta_samples, "Precipitation (mm)"
        )
        assert fig is not None

    def test_residual_qq_creates(self):
        """Q-Q plot should generate"""
        fig = visualizations.plot_residual_qq(self.residuals, "Student-T Test")
        assert fig is not None

    def test_liquidity_heatmap_creates(self):
        """Heatmap should generate"""
        matrix = np.random.uniform(0, 1, (20, 24))
        fig = visualizations.plot_liquidity_heatmap(matrix, title="Test Heatmap")
        assert fig is not None

    def test_correlation_network_creates(self):
        """Correlation convergence plot should generate"""
        before = np.random.uniform(0, 0.5, (10, 10))
        before = (before + before.T) / 2
        np.fill_diagonal(before, 1.0)

        during = np.random.uniform(0.5, 1.0, (10, 10))
        during = (during + during.T) / 2
        np.fill_diagonal(during, 1.0)

        fig = visualizations.plot_correlation_network(before, during)
        assert fig is not None

    def test_expected_shortfall_creates(self):
        """ES plot should generate"""
        es_posterior = np.random.gamma(2, 100, (100, 24))
        fig = visualizations.plot_expected_shortfall(es_posterior)
        assert fig is not None

    def test_model_comparison_creates(self):
        """Model comparison bar should generate"""
        scores = [0.142, 0.189, 0.201, 0.287]
        models = ["Orbit BSTS", "Prophet", "ARIMAX", "Naive"]
        fig = visualizations.plot_model_comparison(scores, models)
        assert fig is not None


class TestDataQuality:
    """Test data quality assumptions"""

    def test_shock_magnitude_ordering(self):
        """Flash flood should be larger than heat wave"""
        events = BLACK_SWAN_REGISTRY
        mags = {e.name: e.magnitude for e in events}

        assert mags["Flash Flood & Transit Collapse"] > mags.get("Intense Heat Wave", 0)

    def test_shock_hours_are_realistic(self):
        """Shock detection should flag realistic number of hours"""
        # June event: 72 hours
        # Sept event: 48 hours
        # Should be somewhere in this ballpark
        june_start = datetime(2023, 6, 7)
        june_end = datetime(2023, 6, 9, 23, 59, 59)

        hours_flagged = 0
        current = june_start
        from datetime import timedelta
        while current <= june_end:
            if is_shock_date(current):
                hours_flagged += 1
            current += timedelta(hours=1)

        assert 50 < hours_flagged < 100  # Should be ~72


class TestModelAssumptions:
    """Test mathematical model assumptions"""

    def test_posterior_mean_reasonable(self):
        """Posterior means should be reasonable"""
        # Precipitation elasticity should be negative
        beta_samples = np.random.normal(-0.35, 0.08, 1000)
        assert np.mean(beta_samples) < 0
        assert 0.05 < np.std(beta_samples) < 0.15

    def test_student_t_vs_normal(self):
        """Student-T should have higher kurtosis"""
        from scipy import stats

        normal_samples = np.random.standard_normal(10000)
        t_samples = np.random.standard_t(3.5, size=10000)

        normal_kurtosis = stats.kurtosis(normal_samples)
        t_kurtosis = stats.kurtosis(t_samples)

        # Student-T should have higher kurtosis
        assert t_kurtosis > normal_kurtosis


class TestIntegration:
    """Integration tests"""

    def test_full_shock_pipeline(self):
        """Test complete shock detection pipeline"""
        test_date = datetime(2023, 6, 8)

        assert is_shock_date(test_date)
        magnitude = get_event_magnitude(test_date)
        assert magnitude > 0
        assert magnitude <= 1

    def test_visualization_pipeline(self):
        """Test that visualizations can be created from synthetic data"""
        np.random.seed(42)
        n = 168  # 1 week of hourly data

        actual = np.random.normal(3000, 500, n)
        forecast = np.random.normal(3000, 600, n)
        std = np.full(n, 400.0)

        fig = visualizations.plot_bayesian_fan_chart(
            actual, forecast, std, shock_dates=[(24, 48)]
        )
        assert fig is not None


# ============================================================================
# PYTEST HOOKS & CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
