import numpy as np
import pandas as pd
from .preprocess import discretise

def get_synthetic_distribution(mu: float,
                               sigma: float,
                               num_bins: int = 8,
                               sample_size: int = 50_000,
                               seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic log-normal samples and discretize them into bins.

    Args:
        mu: Mean of the underlying normal distribution (log-scale).
        sigma: Standard deviation of the underlying normal distribution.
        num_bins: Number of discrete bins (2^n, where n is qubit count).
        sample_size: Number of samples to draw.
        seed: Optional random seed for reproducibility.

    Returns:
        bin_centers: Array of length num_bins giving the midpoint of each bin.
        probabilities: Normalized probability for each bin (sums to 1).
        bin_edges: Array of length num_bins+1 giving bin boundaries.
    """
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=sample_size)
    counts, bin_edges, probs = discretise(samples, num_bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, probs, bin_edges

def get_real_distribution(csv_path: str,
                          num_bins: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load real price data from CSV, compute daily log-returns, and discretize.

    Args:
        csv_path: Path to CSV file with at least columns ['Date', 'Close'].
        num_bins: Number of discrete bins.

    Returns:
        bin_centers: Array of length num_bins giving the midpoint of each bin.
        probabilities: Normalized probability for each bin (sums to 1).
        bin_edges: Array of length num_bins+1 giving bin boundaries.
    """
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date')
    prices = df['Close'].values
    log_returns = np.log(prices[1:] / prices[:-1])
    counts, bin_edges, probs = discretise(log_returns, num_bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, probs, bin_edges
