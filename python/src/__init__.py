"""
Resilience & Liquidity Analysis Package
Phase II: Bayesian Inference and Visualization
"""

from .models import ResilienceModel, MultiStationModel, compute_impulse_response
from .shock_registry import BLACK_SWAN_REGISTRY, get_events_by_month, is_shock_date
from .visualizations import (
    plot_bayesian_fan_chart,
    plot_impulse_response_function,
    plot_posterior_beta_distribution,
    plot_residual_qq,
    plot_liquidity_heatmap,
    plot_correlation_network,
    plot_expected_shortfall,
    plot_model_comparison,
)

__version__ = "0.1.0"
__all__ = [
    "ResilienceModel",
    "MultiStationModel",
    "compute_impulse_response",
    "BLACK_SWAN_REGISTRY",
    "get_events_by_month",
    "is_shock_date",
    "plot_bayesian_fan_chart",
    "plot_impulse_response_function",
    "plot_posterior_beta_distribution",
    "plot_residual_qq",
    "plot_liquidity_heatmap",
    "plot_correlation_network",
    "plot_expected_shortfall",
    "plot_model_comparison",
]
