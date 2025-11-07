import numpy as np

def discretise(samples: np.ndarray, num_bins: int = 8):
 
    counts, bin_edges = np.histogram(samples, bins=num_bins)
    probs = counts / counts.sum()
    return counts, bin_edges, probs

def value_to_basis_index(x: float, bin_edges: np.ndarray) -> int:
  

    idx = np.searchsorted(bin_edges, x, side='right') - 1

    idx = max(0, min(idx, len(bin_edges) - 2))
    return idx

    """
    Map a real value `x` to its histogram/bin index given `bin_edges`.

    - `bin_edges` has length (num_bins + 1), e.g. from np.histogram(...)[1].
      Bins are [edge[i], edge[i+1]) for i=0..num_bins-2; the last bin includes the right edge.
    - We use np.searchsorted(edges, x, side='right') to get where `x` would be inserted
      to keep `edges` sorted. If `x` lies in bin i, this returns i+1, so we subtract 1.
    - We then clamp to [0, num_bins-1] so values outside the edge range fall into the
      first or last bin instead of producing an out-of-range index.

    Examples (edges = [0, 10, 20, 30] → 3 bins: [0,10), [10,20), [20,30]):
      x=7    → 0
      x=10   → 1  (exact interior edge goes to the right bin)
      x=30   → 2  (rightmost edge → last bin)
      x=-5   → 0  (clamped left)
      x=999  → 2  (clamped right)
    """
