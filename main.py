import argparse
import numpy as np
import cupy as cp
import cuquantum
from cuquantum.bindings import custatevec

from GPUStateVector import GPUStateVector
from Circuit_Simulator import CircuitSimulator
from MC_Circuits import MC_Circuits

def main():
    np.random.seed(0)
    np.set_printoptions(linewidth=1000, threshold=np.inf) 
    model_params = {"L":20, "x":0.03} 
    mc_params = {"n_mc": 100, "n_measure": 1, "folder": "data/set_1"}

    mc_circuit = MC_Circuits(model_params, mc_params)

    # mc_circuit.run()

    mc_circuit.run_quantum()

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--x", type=float, default=0.03)
    parser.add_argument("--n_mc", type=int, default=100)
    parser.add_argument("--n_measure", type=int, default=1)
    parser.add_argument("--folder", type=str, default="data/set_1")
    parser.add_argument("--start_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_mode", type=str, choices=["classical", "quantum"], default="quantum")
    args = parser.parse_args()

    np.random.seed(args.seed)
    np.set_printoptions(linewidth=1000, threshold=np.inf)
    model_params = {"L": args.L, "x": args.x}
    mc_params = {
        "n_mc": args.n_mc,
        "n_measure": args.n_measure,
        "folder": args.folder,
        "start_ratio": args.start_ratio,
    }

    mc_circuit = MC_Circuits(model_params, mc_params)
    if args.run_mode == "classical":
        mc_circuit.run()
    else:
        mc_circuit.run_quantum()


if __name__ == '__main__':
    main_cli()




