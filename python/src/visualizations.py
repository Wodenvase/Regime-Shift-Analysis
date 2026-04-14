"""
Analytical Visualizations for Resilience & Liquidity Analysis
8 charts for understanding system shocks and recovery
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Set style globally
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)


def plot_bayesian_fan_chart(actual, forecast_mean, forecast_std, shock_dates=None, title=""):
    """
    Chart 1: Bayesian Fan Chart
    Visualizes 95% credible interval vs actuals during shock periods.
    Inference: Shows exactly when model confidence breaks.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    t = np.arange(len(actual))

    # Credible intervals (90%, 80%, 50%)
    for level, alpha in [(0.05, 0.15), (0.10, 0.25), (0.25, 0.35)]:
        z_score = stats.norm.ppf(1 - level / 2)
        upper = forecast_mean + z_score * forecast_std
        lower = forecast_mean - z_score * forecast_std
        ax.fill_between(t, lower, upper, alpha=alpha, label=f"{100*(1-level):.0f}% CI")

    # Actual values
    ax.plot(t, actual, "k-", linewidth=2.5, label="Actual", zorder=10)

    # Highlight shock periods
    if shock_dates:
        for start, end in shock_dates:
            ax.axvspan(start, end, alpha=0.2, color="red", label="Shock Period")

    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_ylabel("Trip Count (station-hour)", fontsize=11)
    ax.set_title(title or "Bayesian Forecast vs Actuals", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return fig


def plot_impulse_response_function(t_horizon, irf_response, shock_name=""):
    """
    Chart 2: Impulse Response Function (IRF)
    Recovery time (t+n) after a shock. Quantifies "Systemic Inertia".
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t_horizon, irf_response, "o-", linewidth=2.5, markersize=8, color="steelblue")
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_between(t_horizon, 0, irf_response, alpha=0.3, color="steelblue")

    # Mark 50% recovery point
    half_max_idx = np.argmin(np.abs(irf_response - 0.5))
    ax.axvline(x=t_horizon[half_max_idx], color="red", linestyle=":", alpha=0.7)
    ax.text(t_horizon[half_max_idx], 0.5, f"  50% Recovery\n  @ t+{t_horizon[half_max_idx]:.1f}h",
            fontsize=10, color="red", fontweight="bold")

    ax.set_xlabel("Time Since Shock (hours)", fontsize=11)
    ax.set_ylabel("Response Magnitude", fontsize=11)
    ax.set_title(f"Impulse Response Function - {shock_name}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    return fig


def plot_posterior_beta_distribution(beta_samples, regressor_name="Precipitation"):
    """
    Chart 3: Posterior Beta Distribution
    Distribution of "Rain Sensitivity" coefficient.
    Inference: Which neighborhoods are elastic vs inelastic?
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(beta_samples, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")

    # Overlay KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(beta_samples)
    x_range = np.linspace(beta_samples.min(), beta_samples.max(), 200)
    ax.plot(x_range, kde(x_range), "r-", linewidth=2.5, label="KDE")

    # Mean and credible interval
    mean = np.mean(beta_samples)
    ci_lower, ci_upper = np.percentile(beta_samples, [2.5, 97.5])

    ax.axvline(mean, color="green", linestyle="-", linewidth=2, label=f"Mean: {mean:.3f}")
    ax.axvline(ci_lower, color="orange", linestyle="--", alpha=0.7, label=f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    ax.axvline(ci_upper, color="orange", linestyle="--", alpha=0.7)

    ax.set_xlabel(f"{regressor_name} Coefficient (β)", fontsize=11)
    ax.set_ylabel("Posterior Density", fontsize=11)
    ax.set_title(f"Posterior Distribution: {regressor_name} Sensitivity", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    return fig


def plot_residual_qq(residuals, title=""):
    """
    Chart 4: Residual Q-Q Plot
    Validates Student-T vs Normal assumption.
    Inference: Do we need fat-tail modeling?
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Q-Q plot vs Normal
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title("Q-Q Plot vs Normal Distribution", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Histogram with overlay (Normal vs Student-T)
    ax2.hist(residuals, bins=40, density=True, alpha=0.6, color="steelblue", edgecolor="black", label="Residuals")

    # Normal fit
    mu, sigma = residuals.mean(), residuals.std()
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    ax2.plot(x_range, stats.norm.pdf(x_range, mu, sigma), "g-", linewidth=2, label="Normal Fit")

    # Student-T fit
    df, loc, scale = stats.t.fit(residuals)
    ax2.plot(x_range, stats.t.pdf(x_range, df, loc, scale), "r-", linewidth=2, label=f"Student-T Fit (df={df:.1f})")

    ax2.set_xlabel("Residuals", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Distribution Fit Comparison", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_liquidity_heatmap(station_availability_matrix, station_names=None, title=""):
    """
    Chart 5: Liquidity Surface Heatmap
    3D map showing "Stationary Probability" (availability) across NYC.
    Inference: "Liquidity Deserts" where system remains empty >4 hours.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    im = ax.imshow(station_availability_matrix, cmap="RdYlGn", aspect="auto", interpolation="nearest")

    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_ylabel("Station", fontsize=11)
    ax.set_title(title or "Liquidity Surface Heatmap", fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, label="Availability (0-1)")

    # Highlight zeros (liquidity deserts)
    ax.contour(station_availability_matrix == 0, levels=[0.5], colors="red", linewidths=1.5)

    plt.tight_layout()
    return fig


def plot_correlation_network(corr_before, corr_during, station_ids=None, n_top_edges=20):
    """
    Chart 6: Correlation Convergence
    Station correlations before vs during shock.
    Inference: "In a crisis, all correlations go to 1"
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Flatten correlations (upper triangle)
    corr_before_flat = corr_before[np.triu_indices_from(corr_before, k=1)]
    corr_during_flat = corr_during[np.triu_indices_from(corr_during, k=1)]

    # Before shock
    ax1.hist(corr_before_flat, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.axvline(corr_before_flat.mean(), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {corr_before_flat.mean():.3f}")
    ax1.set_xlabel("Correlation Coefficient", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Before Shock", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # During shock
    ax2.hist(corr_during_flat, bins=30, alpha=0.7, color="coral", edgecolor="black")
    ax2.axvline(corr_during_flat.mean(), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {corr_during_flat.mean():.3f}")
    ax2.set_xlabel("Correlation Coefficient", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("During Shock", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Correlation Convergence: Before vs During Systemic Shock",
                 fontsize=14, fontweight="bold", y=1.00)

    plt.tight_layout()
    return fig


def plot_expected_shortfall(unmet_demand_posterior, horizon=30, title=""):
    """
    Chart 7: Expected Shortfall (ES) Plot
    Rolling metric of "Unmet Demand" based on posterior's lower bound.
    Inference: Economic cost of station "stockouts".
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    t = np.arange(len(unmet_demand_posterior))

    # Plot point estimate and 90% confidence band
    mean_es = np.mean(unmet_demand_posterior, axis=0)
    ci_lower = np.percentile(unmet_demand_posterior, 5, axis=0)
    ci_upper = np.percentile(unmet_demand_posterior, 95, axis=0)

    ax.plot(t, mean_es, "r-", linewidth=2.5, label="Expected Shortfall (Mean)", zorder=10)
    ax.fill_between(t, ci_lower, ci_upper, alpha=0.3, color="red", label="90% Confidence Band")

    ax.set_xlabel("Time (hours post-shock)", fontsize=11)
    ax.set_ylabel("Unmet Demand (trips)", fontsize=11)
    ax.set_title(title or "Expected Shortfall: Rolling Unmet Demand", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    return fig


def plot_model_comparison(model_scores, models=["Orbit BSTS", "Facebook Prophet", "Naive Mean"]):
    """
    Chart 8: Model Comparison Bar Chart
    Orbit (BSTS) vs Prophet vs Naive baseline.
    Inference: Accuracy premium from Bayesian priors.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["steelblue", "orange", "gray"]
    bars = ax.bar(models, model_scores, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, model_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f"{score:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("RMSE / MAE (lower is better)", fontsize=11)
    ax.set_title("Model Performance Comparison\n(Extreme Weather Days - Sept Flood)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(model_scores) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    # Calculate improvement
    naive_baseline = model_scores[-1]
    orbit_improvement = (naive_baseline - model_scores[0]) / naive_baseline * 100
    ax.text(0.5, 0.95, f"Bayesian Advantage: {orbit_improvement:.1f}% over Naive",
            transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
            fontsize=11, fontweight="bold")

    return fig
