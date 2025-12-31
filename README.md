# Monte Carlo Quantum Sampling

This project is an algorithm that uses quantum computing results to update in a Monte Carlo 
simulation. 
The algorithm is useful to compute Renyi-2 correlator and other observables for open systems.
Quantum computing part is simulated using cuquantum package.

---
## Class Structure
- `CircuitSimulator` (`Circuit_Simulator.py`)
  - Inherits from `GPUStateVector` and defines the quantum circuit workflow.
  - Builds the ZXZ (or Ising) initialization circuit, appends model gates, and applies them.
  - Manages optional quantum channel operators and correlation operators.
- `CircuitSimulatorPairs` (`Circuit_Simulator_pairs.py`)
  - Variant simulator for pair-based updates and measurements.
  - Shares the same gate-application interface as `CircuitSimulator`.
- `MC_Circuits` (`MC_Circuits.py`)
  - Monte Carlo driver that proposes updates to `quantum_channel_config`.
  - Uses `CircuitSimulator` to evaluate weights and acceptance ratios.
  - Collects measurements and saves basic plots.
  -
  - Two sets of update scheme: Classical (obtaining weight and generate classical number) and Quantum (use Grover algorithm and generate measurement results).
  - Classical: generate Circuit_Simulator for each update proposed
  - Quantum: store one Circuit_Simulator and generate a new Circuit_Simulator to form Circuit_Simulator_pairs for each update propsed
- `GPUSTATEVector`
  - Underlying class to apply quantum gates and measurements using cuquantum.

## Usage
- `CircuitSimulator`
  - Contains `gate_list` (including other gates) and `gate_list_state` (only to generate the state)
  - Initialize to get state gates (both `gate_list` and `gate_list_state`)
  - Add other gates (quantum channel, correlator, state_gates adjoint) to the `gate_list`
  - `apply_gates()` to compute
- `zxz_states`
  - Calculate Renyi-2 correlator by exact MPS contractions
- `Euler_coef_calculation.py` is used to calculate the coefficients of Euler product expansion of a function.
This is useful to correct higher order difference in Grover's algorithm.

## CLI Usage
Run the main script with explicit parameters:

```bash
python -u main.py \
  --L 20 \
  --x 0.03 \
  --n_mc 100 \
  --n_measure 1 \
  --folder data/set_1 \
  --start_ratio 0.25 \
  --run_mode quantum \
  --seed 0
```

Parameter meanings:
- `--L`: system size (number of qubits).
- `--x`: coupling used to set `theta_ZZ`, `theta_X`, `theta_XX` (via `x * pi`).
- `--n_mc`: number of Monte Carlo steps.
- `--n_measure`: measure every `n_measure` steps.
- `--folder`: output directory for plots and saved data.
- `--start_ratio`: fraction of initial measurements to discard before averaging.
- `--run_mode`: `quantum` (Grover-based updates) or `classical`.
- `--seed`: NumPy RNG seed for reproducibility.
