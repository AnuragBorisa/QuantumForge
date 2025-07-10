# tests/test_preprocess.py
import sys, os
# ensure the project root (parent of tests/) is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from preprocess import discretise, value_to_basis_index

def test_discretise_even_split():
    samples = np.array([0.0, 1.0, 2.0, 3.0])
    counts, edges, probs = discretise(samples, num_bins=2)
    assert np.allclose(counts, [2, 2])
    assert np.allclose(edges, [0.0, 1.5, 3.0])
    assert pytest.approx(probs.sum(), rel=1e-6) == 1.0

def test_value_to_basis_index_boundaries():
    edges = np.array([0.0, 2.0, 4.0])
    assert value_to_basis_index(0.0, edges) == 0
    assert value_to_basis_index(1.99, edges) == 0
    assert value_to_basis_index(2.0, edges) == 1
    assert value_to_basis_index(3.5, edges) == 1
    assert value_to_basis_index(-5.0, edges) == 0
    assert value_to_basis_index(10.0, edges) == 1
