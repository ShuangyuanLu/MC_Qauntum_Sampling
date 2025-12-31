from Circuit_Simulator import CircuitSimulator
import numpy as np
import time
from sample_probability import sample_probability
import warnings

class CircuitSimulatorPairs:
    def __init__(self, sim_1, sim_2):
        self.sim_1 = sim_1
        self.sim_2 = sim_2
        self.n_shots = 100 # number of shots for each round of measurement
        self.n_rounds = 10 # number of rounds of measurement to decide update or not
        self.n = sim_1.n_qubits
        self.all_zero_bitstring = "0" * self.n
        self.c_small = 0.3 # small number to avoid over-grovering
        self.M = 2    # number of higher order corrections
        self.n_grover = None

    def grover_rotation_on_both(self):
        self.sim_1.reset()
        self.sim_2.reset()

        self.sim_1.apply_gates()
        self.sim_2.apply_gates()
        p1 = self.sim_1.prob_all_zero()
        p2 = self.sim_2.prob_all_zero()

        # self.n_grover = int(np.floor((self.c_small * np.sqrt(1 / max(p1, p2)) - 1) / 2))
        max_p = 2 ** (- self.n / 2) * 3
        self.n_grover = int(np.floor((self.c_small * np.sqrt(1/ max_p) - 1) / 2))
        # print("n_grover:", self.n_grover)
        self.sim_1.grover_rotation(self.n_grover)
        self.sim_2.grover_rotation(self.n_grover)

        # P1 = self.sim_1.prob_all_zero()
        # P2 = self.sim_2.prob_all_zero()

        # print("p1:", p1, "p2:", p2, "ratio:", p2/p1)
        # print("P1:", P1, "P2:", P2, "ratio after grover:", P2 / P1)

        # r12 = P2 / (P1 + P2 - P1 * P2)
        # for m in range(1, self.M + 1):
        #     r12 = r12 * (1 - self.alpha(self.n_grover, m) * P1 ** m) ** 2
        # print("r1->2", r12)

        # r21 = P1 / (P1 + P2 - P1 * P2)
        # for m in range(1, self.M + 1):
        #     r21 = r21 * (1 - self.alpha(self.n_grover, m) * P2 ** m) ** 2
        # print("r2->1", r21)
        # print("ratio after grover and correction:", r12 / r21, p2/p1)

    def update_or_not(self):
        decision = None
        for i in range(self.n_rounds):
            bit_string_list_1 = self.sim_1.sample_bitstrings_all_zero(self.n_shots)
            bit_string_list_2 = self.sim_2.sample_bitstrings_all_zero(self.n_shots)
            if bit_string_list_1.any():
                if bit_string_list_2.any():
                    idx_1 = np.argmax(bit_string_list_1)
                    idx_2 = np.argmax(bit_string_list_2)
                    decision = idx_2 <= idx_1
                    break
                else:
                    decision = False
                    break
            elif bit_string_list_2.any():
                decision = True
                break

        if decision is None:
            print("undecided")
        
        if decision == True:
            stop = False
            for m in range(1, self.M + 1):
                for _ in range(2):
                    if np.random.rand() < self.alpha(self.n_grover, m) and self.sim_1.sample_bitstrings_all_zero(m).all():
                        decision = False
                        stop = True
                        break
                if stop:
                    break

        return decision
    
    def alpha(self, n, m):
        if m == 1:
            coeff = 2 * n * (n + 1) / 3 / (2 * n + 1) ** 2
        if m == 2:
            coeff = 2 * n * (n + 1) * (17 * n ** 2 + 17 * n + 6) / 45 / (2 * n + 1) ** 4
        return coeff

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


# if __name__ == '__main__':
#     n_qubits = 6
#     sim_1 = CircuitSimulator(n_qubits, seed=1)
#     sim_2 = CircuitSimulator(n_qubits, seed=2)
#     sim_pairs = CircuitSimulatorPairs(sim_1, sim_2)

#     sim_pairs.check_grover_algorithm()
#     print("result:", sim_pairs.check_measurement_first_occur_ratio())

    






















