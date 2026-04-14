"""
Bayesian Structural Time Series (BSTS) Models
Phase II: Modeling and Inference
"""

import numpy as np
import pandas as pd
from orbit.models import LGTFull
from orbit.diagnostics import diagnostics
import warnings

warnings.filterwarnings("ignore")


class ResilienceModel:
    """Wrapper for Orbit BSTS modeling of station liquidity"""

    def __init__(self, data: pd.DataFrame, exogenous_vars: list = None):
        """
        Initialize BSTS model

        Args:
            data: DataFrame with columns [timestamp, value, ...]
            exogenous_vars: List of exogenous regressor column names
        """
        self.data = data
        self.exogenous_vars = exogenous_vars or []
        self.model = None
        self.posterior = None

    def fit(
        self,
        num_warmup: int = 500,
        num_samples: int = 500,
        decompose: bool = True
    ):
        """
        Fit BSTS model using HMC sampling

        Args:
            num_warmup: Burn-in iterations
            num_samples: Post-warmup samples
            decompose: Include seasonality decomposition
        """

        # Prepare data: must have 'ds' (datetime) and 'y' (value) columns
        df = self.data.copy()
        if 'ds' not in df.columns:
            df['ds'] = df.index

        # Initialize Local Global Trend model
        self.model = LGTFull(
            response_col='y',
            date_col='ds',
            regressor_col=self.exogenous_vars if self.exogenous_vars else None,
            decompose=decompose,
            model_name='station_liquidity_bsts'
        )

        # Fit model
        self.posterior = self.model.fit(
            df,
            num_warmup=num_warmup,
            num_samples=num_samples,
            seed=42
        )

        return self.posterior

    def predict(self, periods: int = 24):
        """Generate point and interval forecasts"""
        if self.model is None:
            raise ValueError("Model not yet fitted. Call fit() first.")

        forecaster = self.model.predict(self.posterior, periods=periods)
        return forecaster

    def get_shock_sensitivity(self):
        """Extract posterior distribution of shock sensitivity (beta coefficients)"""
        if self.posterior is None:
            raise ValueError("Model not yet fitted.")

        # Extract from posterior samples
        # This is model-specific; Orbit provides posterior through ArviZ
        return self.posterior

    def diagnostics(self):
        """Run diagnostic checks on fitted model"""
        if self.posterior is None:
            raise ValueError("Model not yet fitted.")

        return diagnostics(self.posterior)


class MultiStationModel:
    """Hierarchical model across multiple stations"""

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: Station-level time series with columns [station_id, ds, y, exog_vars]
        """
        self.data = data
        self.models = {}

    def fit_per_station(self, num_warmup: int = 300, num_samples: int = 300):
        """Fit independent BSTS model for each station"""

        for station_id in self.data['station_id'].unique():
            station_data = self.data[self.data['station_id'] == station_id].copy()
            model = ResilienceModel(station_data)
            model.fit(num_warmup=num_warmup, num_samples=num_samples)
            self.models[station_id] = model

        return self.models


def compute_impulse_response(posterior, shock_magnitude: float = 1.0, horizon: int = 24):
    """
    Compute impulse response function (IRF) post-shock

    Args:
        posterior: Fitted model posterior
        shock_magnitude: Size of shock (e.g., 1 unit precipitation)
        horizon: Time horizon (hours) to track response

    Returns:
        irf: Array of responses at t+0, t+1, ..., t+horizon
    """
    # This would extract shock decay from BSTS level/slope terms
    # Placeholder for more detailed implementation
    irf = np.zeros(horizon)
    # IRF would be exp(-alpha * t) * shock_magnitude for typical mean reversion
    decay_rate = 0.15  # Hours to recover
    irf = shock_magnitude * np.exp(-decay_rate * np.arange(horizon))

    return irf
