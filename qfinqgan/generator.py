import numpy as np
from qiskit import transpile

class QuantumGenerator:

    def __init__(self, num_qubits, ansatz, sim):
        self.num_qubits = num_qubits       
        self.ansatz     = ansatz
        self.sim        = sim 
        self.params     = None 

    def set_parameters(self, theta):
        theta = np.array(theta)
        if theta.shape != (len(self.ansatz.parameters),):
            raise ValueError(
                f"Expected {len(self.ansatz.parameters)} parameters, got {theta.shape}"
            )
        self.params = theta

    def random_parameters(self):
        theta = np.random.uniform(
            0, 2*np.pi, size=len(self.ansatz.parameters)
        )
        self.set_parameters(theta)
        return theta

    def sample(self, shots):
        if self.params is None:
            raise RuntimeError(     
                "Parameters not set; call set_parameters() or random_parameters() first."
            )

        # bind the numeric parameters
        bound_qc = self.ansatz.assign_parameters(self.params)
        # add measurements so memory slots exist
        bound_qc.measure_all()

        qc_t = transpile(bound_qc, self.sim)
        job  = self.sim.run(qc_t, shots=shots, memory=True)
        result     = job.result()
        bitstrings = result.get_memory()

        # convert each "010"→2, etc.
        ints = np.array([int(bs, 2) for bs in bitstrings], dtype=int)
        return ints
