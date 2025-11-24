from GPUStateVector import GPUStateVector
import numpy as np
import cupy as cp
import time
import scipy


class CircuitSimulator(GPUStateVector):
    def __init__(self, n_qubits, seed=None):
        super(CircuitSimulator, self).__init__(n_qubits, seed=seed)
        self.gate_list = []
        self.gate_qubit_list = []
        self.gate_list_state = []
        self.gate_qubit_list_state = []
        self.dtype = cp.complex128

        self.parameters = {"theta": np.pi / 3}

    def initialize(self):
        self.initialize_zxz()
        self.add_gates_zxz()
        self.gate_list = self.gate_list_state.copy()
        self.gate_qubit_list = self.gate_qubit_list_state.copy()
        self.reset()

    def initialize_zxz(self):
        if self.n_qubits % 2 == 1:
            raise ValueError("Number of qubits must be even")
        H = cp.array([[1, 1], [1, -1]], dtype=self.dtype) / cp.sqrt(2)
        CZ = cp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=self.dtype)
        for i in range(self.n_qubits):
            self.gate_list_state.append(H)
            self.gate_qubit_list_state.append(i)
        for i in range(self.n_qubits // 2):
            self.gate_list_state.append(CZ)
            self.gate_qubit_list_state.append((2 * i, 2 * i + 1))
        for i in range(self.n_qubits // 2):
            self.gate_list_state.append(CZ)
            self.gate_qubit_list_state.append((2 * i + 1, (2 * i + 2) % self.n_qubits))

    def add_gates_zxz(self):
        theta = self.parameters["theta"]
        G = scipy.linalg.expm(1j * theta * np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]))
        G = cp.array(G, dtype=self.dtype)
        for i in range(self.n_qubits // 2):
            self.gate_list_state.append(G)
            self.gate_qubit_list_state.append((2 * i, 2 * i + 1))
        for i in range(self.n_qubits // 2):
            self.gate_list_state.append(G)
            self.gate_qubit_list_state.append((2 * i + 1, (2 * i + 2) % self.n_qubits))

    def apply_gates(self):
        for i in range(len(self.gate_qubit_list)):
            qubit = self.gate_qubit_list[i]
            gate = self.gate_list[i]
            if isinstance(qubit, int):
                self.apply_1q_gate(gate, qubit)
            else:
                self.apply_2q_gate(gate, *qubit)

    def apply_gates_adjoint(self):
        for i in range(len(self.gate_qubit_list) - 1, -1, -1):
            qubit = self.gate_qubit_list[i]
            gate = self.gate_list[i]
            if isinstance(qubit, int):
                self.apply_1q_gate(gate, qubit, adjoint=True)
            else:
                self.apply_2q_gate(gate, *qubit, adjoint=True)

    def grover_rotation(self, n):
        for i in range(n):
            self.reflection_all_zero()
            self.apply_gates_adjoint()
            self.reflection_all_zero()
            self.apply_gates()

    def check_zxz(self):
        H = cp.array([[1, 1], [1, -1]], dtype=self.dtype) / cp.sqrt(2)
        self.apply_1q_gate(H, 1)
        zxz = self.expectation_z_string([0, 1, 2])
        print(zxz)

    def print(self):
        for i in range(len(self.gate_list)):
            print(self.gate_qubit_list[i])
            print(self.gate_list[i])


# if __name__ == '__main__':
#     sim_1 = CircuitSimulator(10)
#     sim_1.initialize()
#     sim_1.apply_gates()
#
#     sim_2 = CircuitSimulator(10)
#     sim_2.initialize()
#     sim_2.apply_gates()
#
#
#     print(sim_1.prob_all_zero())
#     # sim.print()




