import numpy as np
import pytest
from qiskit_aer import AerSimulator
from qiskit.circuit.library import RealAmplitudes

from qfinqgan.generator import QuantumGenerator

def test_generator_sample_shape():
  
    num_qubits = 3
    ansatz = RealAmplitudes(num_qubits, reps=1)
    backend =  AerSimulator()
    

    gen = QuantumGenerator(num_qubits, ansatz, backend)
    theta = gen.random_parameters()
    assert isinstance(theta, np.ndarray)
    assert theta.shape == (len(ansatz.parameters),)
    
   
    shots = 1000
    samples = gen.sample(shots)
    
   
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (shots,)
  
    assert samples.min() >= 0
    assert samples.max() < 2**num_qubits
