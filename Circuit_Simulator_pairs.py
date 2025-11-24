from Circuit_Simulator import CircuitSimulator
import numpy as np
import time
from sample_probability import sample_probability
import warnings

class CircuitSimulatorPairs:
    def __init__(self, sim_1, sim_2):
        self.sim_1 = sim_1
        self.sim_2 = sim_2
        self.n_shots = 50
        self.n = sim_1.n_qubits
        self.all_zero_bitstring = "0" * self.n
        self.initialize()

    def initialize(self):
        self.sim_1.initialize()
        self.sim_1.apply_gates()
        self.sim_2.parameters['theta'] = np.pi / 4
        self.sim_2.initialize()
        self.sim_2.apply_gates()

    def update_ratio(self):
        pass

    def update_or_not(self):
        for i in range(10):
            bit_string_list_1 = self.sim_1.sample_bitstrings_all_zero(self.n_shots)
            bit_string_list_2 = self.sim_2.sample_bitstrings_all_zero(self.n_shots)
            if bit_string_list_1.any():
                if bit_string_list_2.any():
                    idx_1 = np.argmax(bit_string_list_1)
                    idx_2 = np.argmax(bit_string_list_2)
                    return idx_2 <= idx_1
                else:
                    return False
            elif bit_string_list_2.any():
                return True
            else:
                warnings.warn("Update not decided.")

        return None

    def check_measurement_first_occur_ratio(self):
        n_overall = 100
        n_update = 0
        for i in range(n_overall):
            if self.update_or_not():
                n_update += 1
        return n_update / n_overall

    def check_grover_algorithm(self):
        p1 = self.sim_1.prob_all_zero()
        print("p1:", p1)

        p2 = self.sim_2.prob_all_zero()
        print("p2:", p2)

        print("ratio:", p2/p1)
        print("prob:", p2 / (p1 + p2 - p1 * p2))


if __name__ == '__main__':
    n_qubits = 6
    sim_1 = CircuitSimulator(n_qubits, seed=1)
    sim_2 = CircuitSimulator(n_qubits, seed=2)
    sim_pairs = CircuitSimulatorPairs(sim_1, sim_2)

    sim_pairs.check_grover_algorithm()
    print("result:", sim_pairs.check_measurement_first_occur_ratio())






















