from GPUStateVector import GPUStateVector
import numpy as np
import cupy as cp
import time
import scipy


class CircuitSimulator(GPUStateVector):
    # Simulator zxz quantum circuit
    def __init__(self, n_qubits, parameters={}, seed=None):
        super(CircuitSimulator, self).__init__(n_qubits, seed=seed)
        self.gate_list = [] #  may add other gates later
        self.gate_qubit_list = []
        self.gate_adjoint_list = []
        self.gate_list_state = [] # only for generating state
        self.gate_qubit_list_state = []
        self.gate_adjoint_list_state = []
        self.dtype = cp.complex128

        self.x = parameters["x"]
        self.model_params = {"theta_ZZ": np.pi * self.x, "theta_X": np.pi * self.x, "theta_XX": np.pi * self.x}
        # self.model_params = {"theta_ZZ": np.pi * 0.0, "theta_X": np.pi * 0.0, "theta_XX": np.pi * 0.05}
        
        self.quantum_channel_operator = []
        self.corr_operator = []
        self.initialize_quantum_channel()

    def initialize_quantum_channel(self):
        # alpha = np.pi / 3
        # self.quantum_channel_operator = [cp.eye(2, dtype=self.dtype), cp.array([[np.cos(alpha), 1j * np.sin(alpha)], [1j * np.sin(alpha), np.cos(alpha)]], dtype=self.dtype), cp.array([[np.cos(alpha), -1j * np.sin(alpha)], [-1j * np.sin(alpha), np.cos(alpha)]], dtype=self.dtype), cp.array([[np.cos(2 * alpha), 1j * np.sin(2 * alpha)], [1j * np.sin(2 * alpha), np.cos(2 * alpha)]], dtype=self.dtype), cp.array([[np.cos(2 * alpha), -1j * np.sin(2 * alpha)], [-1j * np.sin(2 * alpha), np.cos(2 * alpha)]], dtype=self.dtype)]
        alpha = np.pi / 4
        self.quantum_channel_operator = [cp.array([[np.cos(alpha), 1j * np.sin(alpha)], [1j * np.sin(alpha), np.cos(alpha)]], dtype=self.dtype), cp.array([[np.cos(alpha), -1j * np.sin(alpha)], [-1j * np.sin(alpha), np.cos(alpha)]], dtype=self.dtype)]
        # self.quantum_channel_operator = [cp.array([[1, 0], [0, 1]], dtype=self.dtype), cp.array([[0, 1], [1, 0]], dtype=self.dtype)]
        # self.quantum_channel_operator = [cp.array([[1, 0], [0, 1]], dtype=self.dtype), cp.array([[1, 0], [0, -1]], dtype=self.dtype)]
        self.corr_operator = cp.array([[1, 0], [0, -1]], dtype=self.dtype)


    def initialize(self):
        self.initialize_zxz()
        # self.initialize_ising()
        self.gate_list = self.gate_list_state.copy()
        self.gate_qubit_list = self.gate_qubit_list_state.copy()
        self.gate_adjoint_list = self.gate_adjoint_list_state.copy()
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
        for i in range(self.n_qubits // 2 - 1):
            self.gate_list_state.append(CZ)
            self.gate_qubit_list_state.append((2 * i + 1, 2 * i + 2))
        self.gate_adjoint_list_state = [False] * len(self.gate_list_state)

        self.add_gates_zxz()

    def add_gates_zxz(self):
        theta_ZZ = self.model_params["theta_ZZ"] 
        theta_X = self.model_params["theta_X"]
        theta_XX = self.model_params["theta_XX"]
        ZZ = np.kron(np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]])) 
        Z = np.kron(np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, 1]])) + np.kron(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, -1]])) 
        X = np.kron(np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])) + np.kron(np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]]))
        G = scipy.linalg.expm(1j * theta_ZZ * ZZ) @ scipy.linalg.expm(1j * theta_X * X) 
        G = cp.array(G, dtype=self.dtype)

        XX = np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]))
        G_XX = scipy.linalg.expm(1j * theta_XX * XX) # scipy.linalg.expm(1j * theta_X * X)
        G_XX = cp.array(G_XX, dtype=self.dtype)
        for i in range(self.n_qubits//2):
            self.gate_list_state.append(G_XX)
            self.gate_qubit_list_state.append((2 * i, 2 * i + 1))
        for i in range(self.n_qubits//2 -1):
            self.gate_list_state.append(G_XX)
            self.gate_qubit_list_state.append((2 * i +1, 2 * i +2))

        for i in range(0, self.n_qubits // 2 - 1, 2):
            self.gate_list_state.append(G)
            self.gate_qubit_list_state.append((2 * i, 2 * i + 2))
        for i in range(1, self.n_qubits // 2 - 1, 2):
            self.gate_list_state.append(G)
            self.gate_qubit_list_state.append((2 * i, 2 * i + 2))
        for i in range(0, self.n_qubits // 2 - 1, 2):
            self.gate_list_state.append(G)
            self.gate_qubit_list_state.append((2 * i + 1, 2 * i + 3))
        for i in range(1, self.n_qubits // 2 - 1, 2):
            self.gate_list_state.append(G)
            self.gate_qubit_list_state.append((2 * i + 1, 2 * i + 3))

        self.gate_adjoint_list_state = [False] * len(self.gate_list_state)

    def initialize_ising(self):
        theta_ZZ = self.model_params["theta_ZZ"] 
        theta_X = self.model_params["theta_X"]
        theta_XX = self.model_params["theta_XX"]
        
        H = cp.array([[1, 1], [1, -1]], dtype=self.dtype) / cp.sqrt(2)
        for i in range(self.n_qubits):
            self.gate_list_state.append(H)
            self.gate_qubit_list_state.append(i)

        ZZ = np.kron(np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]])) 
        X = np.kron(np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])) + np.kron(np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]]))
        XX = np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]))
        G = scipy.linalg.expm(1j * theta_X /2 * X) @ scipy.linalg.expm(1j * theta_ZZ * ZZ) @ scipy.linalg.expm(1j * theta_X /2 * X)
        G = cp.array(G, dtype=self.dtype)
        for i in range(self.n_qubits // 2):
            self.gate_list_state.append(G)
            self.gate_qubit_list_state.append((2 * i, 2 * i + 1))
        for i in range(self.n_qubits // 2 - 1):
            self.gate_list_state.append(G)
            self.gate_qubit_list_state.append((2 * i + 1, 2 * i + 2))
        # for i in range(self.n_qubits // 2):
        #     self.gate_list_state.append(G)
        #     self.gate_qubit_list_state.append((2 * i, 2 * i + 1))
        self.gate_adjoint_list_state = [False] * len(self.gate_list_state)

    def add_quantum_channel_gate(self, quantum_channel_config):
        for i in range(self.n_qubits):
            if quantum_channel_config[i] != -1:
                self.gate_list.append(self.quantum_channel_operator[quantum_channel_config[i]])
                self.gate_qubit_list.append(i)
                self.gate_adjoint_list.append(False)

    def add_gates_state_adjoint(self):
        # Append the adjoint gates in reverse order of application
        self.gate_qubit_list += self.gate_qubit_list_state[::-1]
        self.gate_list += [gate for gate in self.gate_list_state[::-1]]
        self.gate_adjoint_list += [True] * len(self.gate_qubit_list_state)

    def add_corr_gate(self, i: int, j: int):
        self.gate_list += [self.corr_operator, self.corr_operator]
        self.gate_qubit_list += [i, j]
        self.gate_adjoint_list += [False, False]

    def apply_gates(self):
        for i in range(len(self.gate_qubit_list)):
            qubit = self.gate_qubit_list[i]
            gate = self.gate_list[i]
            adjoint = self.gate_adjoint_list[i]
            if isinstance(qubit, int):
                self.apply_1q_gate(gate, qubit, adjoint=adjoint)
            else:
                self.apply_2q_gate(gate, *qubit, adjoint=adjoint)

    def apply_gates_adjoint(self):
        for i in range(len(self.gate_qubit_list) - 1, -1, -1):
            qubit = self.gate_qubit_list[i]
            gate = self.gate_list[i]
            adjoint = self.gate_adjoint_list[i]
            if isinstance(qubit, int):
                self.apply_1q_gate(gate, qubit, adjoint= not adjoint)
            else:
                self.apply_2q_gate(gate, *qubit, adjoint= not adjoint)

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
            #print(self.gate_list[i])



# if __name__ == '__main__':
#     sim_1 = CircuitSimulator(10, parameters={"x":0.03})
#     sim_1.initialize()
#     sim_1.apply_gates()

#     sim_1.print()



#     sim_2 = CircuitSimulator(10)
#     sim_2.initialize()
#     sim_2.apply_gates()
#
#
#     print(sim_1.prob_all_zero())
#     # sim.print()





