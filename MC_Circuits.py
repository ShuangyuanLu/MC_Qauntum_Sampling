from Circuit_Simulator import CircuitSimulator
from Circuit_Simulator_pairs import CircuitSimulatorPairs
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import shelve
from timing_utils import timecall
from mc_average import mc_average
from Circuit_Simulator_pairs import CircuitSimulatorPairs


class MC_Circuits:
    def __init__(self, model_params, mc_params):
        self.mc_params = mc_params
        self.model_params = model_params
        self.L = model_params.get("L", None)
        self.n_mc = mc_params.get("n_mc", 1000)
        self.n_measure = mc_params.get("n_measure", 50)
        self.folder = mc_params.get("folder", "data")

        self.quantum_channel_config = np.zeros(self.L, dtype=int)
        self.quantum_channel_config[1::2] = -1
        # self.quantum_channel_config = np.random.randint(0, 2, size=self.L)

        self.sim_quantum = None
        self.n_p1_min = 10

        self.measurements = {"correlation": [], "channel_density": []}
        self.weight = None
        self.n_accept = 0

        self.p = 0.5
        self.start_ratio = mc_params.get("start_ratio", 1 / 4)
        # self.quantum_channel_n = 5
        # self.quantum_channel_p = [1 + 8/3 * (self.p ** 2 - self.p), 2 * 2/3 * self.p * (1 - 4/3 * self.p), 2 * 2/3 * self.p * (1 - 4/3 * self.p), 4 / 9 * self.p ** 2, 4 / 9 * self.p ** 2]
        self.quantum_channel_n = 2
        self.quantum_channel_p = [0.5, 0.5]
    
    @timecall
    def run(self):
        self.weight = self.get_weight()
        for i_mc in range(self.n_mc):
            # for site_update in range(0, self.L, 2):
            for _ in range(0, self.L, 2):
                site_update = np.random.randint(0, self.L//2) * 2
                self.update(site_update)
            if i_mc % self.n_measure == self.n_measure - 1:
                self.measure()
            if i_mc % (self.n_mc // 10) == (self.n_mc // 10 - 1):
                print(np.round(i_mc / self.n_mc * 100, 2), "% done")
        print("Acceptance ratio:", self.n_accept / self.n_mc)   
        self.save_data()

    def update(self, site_update):
        weight_old = self.get_weight()
        spin_original = self.quantum_channel_config[site_update]
        spin_updated = (np.random.randint(1, self.quantum_channel_n) + spin_original) % self.quantum_channel_n
        # print(self.quantum_channel_config)
        # print("Propose to flip sites:", site_update)
        # print("From", spin_original, "to", spin_updated)
        self.quantum_channel_config[site_update] = spin_updated
        weight_new = self.get_weight()
        ratio = weight_new / weight_old * self.quantum_channel_p[spin_updated] / self.quantum_channel_p[spin_original]
        # print("ratio:", ratio, weight_old, weight_new)

        if ratio < np.random.rand():
            # print("\033[31mreject\033[0m")
            self.quantum_channel_config[site_update] = spin_original
        else:
            # print("\033[32maccept\033[0m")
            # print("ratio:", ratio, weight_old, weight_new)
            self.weight = weight_new
            self.n_accept += 1

    def get_weight(self, corr_operator_included: bool = False, qubit_i=1, qubit_j=3):
        sim = CircuitSimulator(self.L, parameters=self.model_params)
        sim.initialize()
        sim.add_quantum_channel_gate(self.quantum_channel_config)
        if corr_operator_included:
            sim.add_corr_gate(qubit_i, qubit_j)
        sim.add_gates_state_adjoint()
        sim.apply_gates()
        
        weight = sim.prob_all_zero()
        #print("Weight:", weight)
        return weight

    def measure(self):
        self.measurements["channel_density"].append(np.sum(self.quantum_channel_config[0::2]) * 2 / self.L)
        self.measurements["correlation"].append(self.measure_correlation(self.L // 8 * 2 + 1, self.L - self.L // 8 * 2 - 1))
        
    def measure_correlation(self, qubit_i, qubit_j):
        weight = self.weight
        weight_new = self.get_weight(corr_operator_included=True, qubit_i=qubit_i, qubit_j=qubit_j)
        correlation = weight_new / weight
        # print(weight, weight_new, correlation)
        return correlation
    
    @timecall
    def run_quantum(self):
        self.initialize_quantum()

        for i_mc in range(self.n_mc):
            for site_update in range(0, self.L, 2):
                self.update_quantum(site_update)
            if i_mc % self.n_measure == self.n_measure - 1:
                self.measure_quantum()
            if i_mc % (self.n_mc // 10) == (self.n_mc // 10 - 1):
                print(np.round(i_mc / self.n_mc * 100, 2), "% done")
        print("Acceptance ratio:", self.n_accept / self.n_mc)   
        self.save_data()
    
    def initialize_quantum(self):
        self.sim_quantum = CircuitSimulator(self.L, parameters=self.model_params)
        self.sim_quantum.initialize()
        self.sim_quantum.add_quantum_channel_gate(self.quantum_channel_config)
        self.sim_quantum.add_gates_state_adjoint()

    def update_quantum(self, site_update):
        spin_updated = self.quantum_channel_config.copy()
        spin_updated[site_update] = (np.random.randint(1, self.quantum_channel_n) + spin_updated[site_update]) % self.quantum_channel_n

        # print("Propose to flip sites:", site_update)
        # print("From", self.quantum_channel_config[site_update], "to", spin_updated[site_update])

        sim_new = CircuitSimulator(self.L, parameters=self.model_params)
        sim_new.initialize()
        sim_new.add_quantum_channel_gate(spin_updated)
        sim_new.add_gates_state_adjoint() 

        # print("spin_config:", self.quantum_channel_config)
        # print("new__config:", spin_updated)

        sim_pair = CircuitSimulatorPairs(self.sim_quantum, sim_new)
        sim_pair.grover_rotation_on_both()

        update_or_not = sim_pair.update_or_not()
        # update_or_not = self.check_update_quantum_possibility(sim_pair)

        # print("Update or not:", update_or_not)

        if update_or_not:
            self.sim_quantum = sim_new
            self.quantum_channel_config = spin_updated
            self.n_accept += 1

        # print("spin_config:", self.quantum_channel_config)

    def measure_correlation_quantum(self, qubit_i, qubit_j):
        sim_new = CircuitSimulator(self.L, parameters=self.model_params)
        sim_new.initialize()
        sim_new.add_quantum_channel_gate(self.quantum_channel_config)
        sim_new.add_corr_gate(qubit_i, qubit_j)
        sim_new.add_gates_state_adjoint()

        sim_pair = CircuitSimulatorPairs(sim_new, self.sim_quantum)
        sim_pair.grover_rotation_on_both()

        n1 = 0
        n2 = 0
        N = 0
        while n1 < self.n_p1_min:
            update_or_not = sim_pair.update_or_not()
            if update_or_not == True:
                n1 += 1
            N += 1

        sim_pair.sim_1, sim_pair.sim_2 = sim_pair.sim_2, sim_pair.sim_1
        for _ in range(N):
            update_or_not = sim_pair.update_or_not()
            if update_or_not == True:
                n2 += 1

        # print(n1, n2, N)
        correlation = n2 / n1
        return correlation

    def measure_quantum(self):
        self.measurements["channel_density"].append(np.sum(self.quantum_channel_config[0::2]) * 2 / self.L)
        self.measurements["correlation"].append(self.measure_correlation_quantum(self.L // 8 * 2 + 1, self.L - self.L // 8 * 2 - 1))

    def save_data(self):
        data_dir = self.folder
        os.makedirs(data_dir, exist_ok=True)

        plt.plot(self.measurements["channel_density"])
        plt.savefig(os.path.join(data_dir, "channel_density.png"))
        plt.clf()
        plt.plot(self.measurements["correlation"])
        plt.savefig(os.path.join(data_dir, "correlation.png"))
        plt.clf()
        n_measurements = len(self.measurements["channel_density"])
        n_start = int(n_measurements * self.start_ratio)
        chennel_density_average, channel_density_std, channel_density_tau, _, _ = mc_average(np.array(self.measurements["channel_density"][n_start:]))
        correlation_average, correlation_std, correlation_tau, _, _ = mc_average(np.array(self.measurements["correlation"][n_start:]))
        acceptance_ratio = self.n_accept / self.n_mc/ self.n_measure if self.n_mc else 0.0

        shelf_path = os.path.join(data_dir, "measurements.db")
        with shelve.open(shelf_path) as db:
            db["channel_density"] = np.array(self.measurements["channel_density"])
            db["correlation"] = np.array(self.measurements["correlation"])
            db["acceptance_ratio"] = acceptance_ratio
            db["model_params"] = self.model_params
            db["mc_params"] = self.mc_params
            db["p"] = self.p
            db["start_ratio"] = self.start_ratio
            db["channel_density_average"] = chennel_density_average
            db["channel_density_std"] = channel_density_std
            db["channel_density_tau"] = channel_density_tau
            db["correlation_average"] = correlation_average
            db["correlation_std"] = correlation_std
            db["correlation_tau"] = correlation_tau
        print("Average channel density:", chennel_density_average, channel_density_std, channel_density_tau)
        print("Average correlation:", correlation_average, correlation_std, correlation_tau)

    # def update_neighbor(self):
    #     weight_old = self.get_weight()
    #     site_update = np.random.randint(self.L - 1)
    #     self.quantum_channel_config[site_update] ^= 1
    #     self.quantum_channel_config[site_update + 1] ^= 1
    #     weight_new = self.get_weight()
    #     ratio = weight_new / weight_old

    #     if ratio < np.random.rand():
    #         self.quantum_channel_config[site_update] ^= 1
    #         self.quantum_channel_config[site_update + 1] ^= 1
    #     else:
    #         # print("ratio:", ratio, weight_old, weight_new)
    #         self.weight = weight_new
    #         self.n_accept += 1

    def check_measure_correlator_quantum(self):
        self.weight = self.get_weight()
        classical_correlator = self.measure_correlation(self.L // 8 * 2 + 1, self.L - self.L // 8 * 2 - 1)
        print("Classical correlator:", classical_correlator)
        
        self.initialize_quantum()
        N = 1000
        correlator_list = []
        for i in range(N):
            correlator = self.measure_correlation_quantum(self.L // 8 * 2 + 1, self.L - self.L // 8 * 2 - 1)
            correlator_list.append(correlator)

        print("Average correlator:", np.mean(correlator_list), "Std:", np.std(correlator_list) / np.sqrt(N))           
    
    def check_update_quantum_possibility(self, sim_pair):
        N = 100000
        update_count = 0
        for i in range(N):
            update_or_not = sim_pair.update_or_not()
            if update_or_not:
                update_count += 1
        ratio = update_count / N
        print("Estimated update ratio:", ratio)
        return update_or_not

    @timecall
    def check_nonzero_elements_even(self, n_spin: int = 5):
        if n_spin < 2:
            raise ValueError("n_spin must be >= 2")
        even_sites = [i for i in range(self.L) if i % 2 == 0]
        n_even = len(even_sites)
        n = 0
        bins = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]
        bin_counts = [0] * (len(bins) - 1)
        total_configs = n_spin ** n_even
        for spin in range(total_configs):
            spin_config_even = np.zeros(n_even, dtype=int)
            tmp = spin
            for idx in range(n_even - 1, -1, -1):
                spin_config_even[idx] = tmp % n_spin
                tmp //= n_spin
            spin_config_full = np.zeros(self.L, dtype=int)
            spin_config_full[1::2] = -1
            spin_config_full[even_sites] = spin_config_even
            self.quantum_channel_config = spin_config_full
            weight = self.get_weight()
            print(weight)
            if weight > 1e-6:
                n += 1
            for i in range(len(bin_counts)):
                if bins[i] >= weight > bins[i + 1]:
                    bin_counts[i] += 1
                    break
        print("Number of non-zero elements (even sites only):", n, "out of", total_configs, n / total_configs * n_even)
        for i in range(len(bin_counts)):
            upper = bins[i]
            lower = bins[i + 1]
            ratio = bin_counts[i] / total_configs
            print(f"Weight in ({lower:.0e}, {upper:.0e}]: {bin_counts[i]} / {total_configs} = {ratio:.6f}")
    

    def check_nonzero_elements_even_with_corr(self, n_spin: int = 5):
        if n_spin < 2:
            raise ValueError("n_spin must be >= 2")
        even_sites = [i for i in range(self.L) if i % 2 == 0]
        n_even = len(even_sites)
        n = 0
        bins = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]
        bin_counts = [0] * (len(bins) - 1)
        new_bin_counts = [0] * (len(bins) - 1)
        ratio_bins = [1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]
        ratio_bin_counts = [0] * (len(ratio_bins) - 1)
        ratio_total = 0
        total_configs = n_spin ** n_even
        for spin in range(total_configs):
            spin_config_even = np.zeros(n_even, dtype=int)
            tmp = spin
            for idx in range(n_even - 1, -1, -1):
                spin_config_even[idx] = tmp % n_spin
                tmp //= n_spin
            spin_config_full = np.zeros(self.L, dtype=int)
            spin_config_full[1::2] = -1
            spin_config_full[even_sites] = spin_config_even
            self.quantum_channel_config = spin_config_full
            weight = self.get_weight()
            # spin_config_full[2: self.L - 1:2] ^= 1
            # self.quantum_channel_config = spin_config_full
            new_weight = self.get_weight(corr_operator_included=True, qubit_i=1, qubit_j=self.L-3)
            # print(weight, new_weight)
            if weight < 1e-6 and new_weight > 1e-6:
                print(weight, new_weight)
                n += 1
            for i in range(len(bin_counts)):
                if bins[i] >= weight > bins[i + 1]:
                    bin_counts[i] += 1
                    break
            for i in range(len(new_bin_counts)):
                if bins[i] >= new_weight > bins[i + 1]:
                    new_bin_counts[i] += 1
                    break
            if weight > 0:
                ratio = new_weight / weight
                print(ratio)
                ratio_total += 1
                for i in range(len(ratio_bin_counts)):
                    if ratio_bins[i] >= ratio > ratio_bins[i + 1]:
                        ratio_bin_counts[i] += 1
                        break
            # print(weight, new_weight/weight)
        print("Number of non-zero elements (even sites only):", n, "out of", total_configs, n / total_configs * n_even)
        for i in range(len(bin_counts)):
            upper = bins[i]
            lower = bins[i + 1]
            ratio = bin_counts[i] / total_configs
            print(f"Weight in ({lower:.0e}, {upper:.0e}]: {bin_counts[i]} / {total_configs} = {ratio:.6f}")
        for i in range(len(new_bin_counts)):
            upper = bins[i]
            lower = bins[i + 1]
            ratio = new_bin_counts[i] / total_configs
            print(f"New weight in ({lower:.0e}, {upper:.0e}]: {new_bin_counts[i]} / {total_configs} = {ratio:.6f}")
        if ratio_total > 0:
            for i in range(len(ratio_bin_counts)):
                upper = ratio_bins[i]
                lower = ratio_bins[i + 1]
                ratio = ratio_bin_counts[i] / ratio_total
                print(f"New/weight in ({lower:.0e}, {upper:.0e}]: {ratio_bin_counts[i]} / {ratio_total} = {ratio:.6f}")

    # def check_nonzero_elements(self):
    #     n=0
    #     bins = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]
    #     bin_counts = [0] * (len(bins) - 1)
    #     for spin in range(2 ** self.L):
    #         spin_config = np.array(list(format(spin, '0{}b'.format(self.L))), dtype=int)
    #         self.quantum_channel_config = spin_config
    #         weight = self.get_weight()
    #         if weight > 1e-6:
    #             n += 1
    #         for i in range(len(bin_counts)):
    #             if bins[i] >= weight > bins[i + 1]:
    #                 bin_counts[i] += 1
    #                 break
    #     print("Number of non-zero elements:", n, "out of", 2 ** self.L, n / 2 ** self.L * self.L)
    #     total_configs = 2 ** self.L
    #     for i in range(len(bin_counts)):
    #         upper = bins[i]
    #         lower = bins[i + 1]
    #         ratio = bin_counts[i] / total_configs
    #         print(f"Weight in ({lower:.0e}, {upper:.0e}]: {bin_counts[i]} / {total_configs} = {ratio:.6f}")


# if __name__ == '__main__':
#     np.random.seed(4)
#     np.set_printoptions(linewidth=1000, threshold=np.inf) 
#     model_params = {"L":20, "x":0.03} 
#     mc_params = {"n_mc": 100, "n_measure": 1}

#     mc_circuit = MC_Circuits(model_params, mc_params)

    # mc_circuit.check_nonzero_elements()
    # mc_circuit.check_nonzero_elements_even(n_spin=2)
    # mc_circuit.check_nonzero_elements_even_with_corr()

    # mc_circuit.quantum_channel_config[3] = 1
    # mc_circuit.quantum_channel_config[4] = 1
    # mc_circuit.quantum_channel_config[8] = 1
    # mc_circuit.quantum_channel_config[5] = 1


    # print("weight:", mc_circuit.get_weight())
    # print("weight with corr:", mc_circuit.get_weight(corr_operator_included=True, qubit_i=1, qubit_j=3)/mc_circuit.get_weight())
    
    # print("ave:", 1 / 2 ** (model_params["L"] // 2))
    # mc_circuit.run()

    # mc_circuit.initialize_quantum()
    # mc_circuit.update_quantum(2)

    # mc_circuit.run_quantum()

    # mc_circuit.check_measure_correlator_quantum()

    # print(mc_circuit.quantum_channel_config)
    # print("Final weight:", mc_circuit.weight)





