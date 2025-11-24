# Monte Carlo Quantum Sampling
This project is an algorithm that uses quantum computing results to update in a Monte Carlo 
simulation. 
The algorithm is useful to compute Renyi-2 correlator and other observables for open systems.
Quantum computing part is simulated using cuquantum package.
---

# Structure
- MC_simulation 
    - Local circuit generator --- models
        - CircuitSimulator --- cuquantum simulation 
        - bernoulli factory

# Usage
- `Euler_coef_calculation.py` is used to calculate the coefficients of Euler product expansion of a function.
This is useful to correct higher order difference in Grover's algorithm.
- 


