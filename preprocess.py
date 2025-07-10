import numpy as np

def discretise(samples: np.ndarray, num_bins: int = 8):
 
    counts, bin_edges = np.histogram(samples, bins=num_bins)
    probs = counts / counts.sum()
    return counts, bin_edges, probs

def value_to_basis_index(x: float, bin_edges: np.ndarray) -> int:
  

    idx = np.searchsorted(bin_edges, x, side='right') - 1

    idx = max(0, min(idx, len(bin_edges) - 2))
    return idx
