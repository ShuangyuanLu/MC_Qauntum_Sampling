import numpy as np
import cupy as cp
import cuquantum
from cuquantum.bindings import custatevec
import time


class GPUStateVector:
    """
    Minimal wrapper around cuStateVec for state-vector simulation.
    - Uses CuPy for storage.
    - Supports 1q/2q gates, expectation of diagonal Z-strings, and full-basis measurement.
    """

    def __init__(self, n_qubits, dtype=cp.complex128, seed=None):
        self.n_qubits = int(n_qubits)
        self.dim = 1 << self.n_qubits
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)

        if self.dtype == cp.complex64:
            self.sv_dtype = cuquantum.cudaDataType.CUDA_C_32F
            self.compute_type = cuquantum.ComputeType.COMPUTE_32F
        elif self.dtype == cp.complex128:
            self.sv_dtype = cuquantum.cudaDataType.CUDA_C_64F
            self.compute_type = cuquantum.ComputeType.COMPUTE_64F
        else:
            raise ValueError("dtype must be cupy.complex64 or cupy.complex128")

        # Allocate |0...0> on GPU
        self.state = cp.zeros(self.dim, dtype=self.dtype)
        self.state[0] = 1.0

        # cuStateVec handle
        self.handle = custatevec.create()

    # ---- core helper ----
    def _apply_matrix(self, gate, targets, adjoint=False, controls=None, control_values=None):
        """
        Internal low-level wrapper for custatevec.apply_matrix.
        gate: CuPy array on GPU, shape (2^k, 2^k)
        targets: iterable of qubit indices (Python ints)
        controls: iterable of control qubits (or None)
        control_values: iterable of 0/1 for each control (or None)
        """
        # Ensure gate is on GPU with correct dtype
        if not isinstance(gate, cp.ndarray):
            gate = cp.asarray(gate, dtype=self.dtype)
        elif gate.dtype != self.dtype:
            gate = gate.astype(self.dtype)

        sv_ptr = int(self.state.data.ptr)
        gate_ptr = int(gate.data.ptr)

        # Matrix dtype (same as state)
        if self.dtype == cp.complex64:
            matrix_dtype = cuquantum.cudaDataType.CUDA_C_32F
        else:
            matrix_dtype = cuquantum.cudaDataType.CUDA_C_64F

        # Targets / controls
        targets = list(map(int, targets))
        n_targets = len(targets)

        if controls is None:
            controls = []
        else:
            controls = list(map(int, controls))

        n_controls = len(controls)

        if control_values is None:
            control_values = [1] * n_controls  # default: all 1
        else:
            control_values = list(map(int, control_values))

        # For the Python bindings, targets/controls/control_values can be Python sequences
        adjoint_flag = 1 if adjoint else 0

        # Workspace query
        workspace_size = custatevec.apply_matrix_get_workspace_size(
            self.handle,
            self.sv_dtype,
            self.n_qubits,
            gate_ptr,
            matrix_dtype,
            custatevec.MatrixLayout.ROW,
            adjoint_flag,
            n_targets,
            n_controls,
            self.compute_type,
        )

        if workspace_size > 0:
            workspace = cp.cuda.alloc(workspace_size)
            workspace_ptr = int(workspace.ptr)
        else:
            workspace_ptr = 0

        custatevec.apply_matrix(
            self.handle,
            sv_ptr,
            self.sv_dtype,
            self.n_qubits,
            gate_ptr,
            matrix_dtype,
            custatevec.MatrixLayout.ROW,
            adjoint_flag,
            targets,
            n_targets,
            controls,
            control_values,
            n_controls,
            self.compute_type,
            workspace_ptr,
            workspace_size,
        )

        # Make sure all work is finished before we read state on host
        cp.cuda.Stream.null.synchronize()

    # ---- public API ----

    def apply_1q_gate(self, gate, qubit, adjoint=False):
        """Apply a 1-qubit gate on qubit index."""
        self._apply_matrix(gate, targets=[qubit], adjoint=adjoint)

    def apply_2q_gate(self, gate, qubit1, qubit2, adjoint=False,
                      controls=None, control_values=None):
        """Apply a 2-qubit gate on (qubit1, qubit2)."""
        self._apply_matrix(
            gate,
            targets=[qubit1, qubit2],
            adjoint=adjoint,
            controls=controls,
            control_values=control_values,
        )

    def reflection_all_zero(self):
        self.state[0] = - self.state[0]

    def prob_all_zero(self):
        """
        Probability of the all-zero computational basis state |00...0>.
        This is simply |psi[0]|^2.
        """
        amp0 = self.state[0]          # complex scalar on GPU
        return float(cp.abs(amp0)**2)  # convert to Python float

    def probabilities(self):
        """Return |psi|^2 as a NumPy array (host)."""
        probs = cp.abs(self.state) ** 2
        return probs.get()

    def sample(self, n_shots=1, rng=None):
        """
        Sample bitstrings in the computational basis using classical sampling from |psi|^2.
        Returns a NumPy array of shape (n_shots,) of integers in [0, 2^n).
        """
        probs = self.probabilities()
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(self.dim, size=n_shots, p=probs)

    def sample_bitstrings(self, n_shots=1, rng=None):
        """
        Same as sample(), but returns bitstrings as strings '0101...'
        """
        samples = self.sample(n_shots=n_shots, rng=rng)
        return [format(s, f"0{self.n_qubits}b") for s in samples]

    def sample_bitstrings_all_zero(self, n_shots=1):
        p = self.prob_all_zero()
        random_bool = self.rng.random(n_shots) < p
        return random_bool

    def expectation_z_string(self, qubits):
        """
        Compute ⟨Z_{q0} Z_{q1} ...⟩ for a set of qubit indices (diagonal in Z basis).
        This is done on CPU using probabilities; fine for small n / debugging.
        """
        qubits = list(map(int, qubits))
        probs = self.probabilities()  # length 2^n

        # For each basis state i, eigenvalue is product over qubits of (+1 for 0, -1 for 1)
        exp_val = 0.0
        for i, p in enumerate(probs):
            if p == 0.0:
                continue
            bitstring = i
            eigen = 1.0
            for q in qubits:
                bit = (bitstring >> q) & 1
                eigen *= 1.0 if bit == 0 else -1.0
            exp_val += eigen * p
        return exp_val

    def get_statevector(self):
        """Return the full statevector as a NumPy array (host)."""
        return self.state.get()

    def reset(self):
        """Reset to |0...0⟩."""
        self.state.fill(0)
        self.state[0] = 1.0
        cp.cuda.Stream.null.synchronize()

    def __del__(self):
        try:
            if hasattr(self, "handle") and self.handle is not None:
                custatevec.destroy(self.handle)
        except Exception:
            # avoid errors on interpreter shutdown
            pass


# if __name__ == "__main__":
#     # Simple test: 2-qubit Bell state (|00> + |11>)/sqrt(2)
#     n_qubits = 2
#     sim = GPUStateVector(n_qubits=n_qubits, dtype=cp.complex64)
#
#     # Define gates on GPU
#     H = cp.array([[1, 1],
#                   [1, -1]], dtype=cp.complex64) / cp.sqrt(2)
#
#     # CNOT in computational basis, control=0, target=1
#     CNOT = cp.array([[1, 0, 0, 0],
#                      [0, 0, 0, 1],
#                      [0, 0, 1, 0],
#                      [0, 1, 0, 0]], dtype=cp.complex64)
#
#     # Start from |00>
#     sim.reset()
#
#     # Apply H on qubit 0
#     sim.apply_1q_gate(H, qubit=0)
#     print(sim.get_statevector())
#
#     # Apply CNOT(0 -> 1)
#     sim.apply_2q_gate(CNOT, qubit1=0, qubit2=1)
#
#     # Check the statevector
#     psi = sim.get_statevector()
#     print("Statevector (host):")
#     print(psi)
#
#     # Expected: approx [1/sqrt(2), 0, 0, 1/sqrt(2)]
#     print("Norm:", np.linalg.norm(psi))
#
#     # Probabilities
#     probs = sim.probabilities()
#     print("Probabilities:", probs)
#
#     # Expectation of Z0 Z1 (should be ~ +1 for Bell state (|00>+|11>)/√2)
#     exp_zz = sim.expectation_z_string([0, 1])
#     print("<Z0 Z1> ~", exp_zz)
#
#     # Sampling
#     bitstrings = sim.sample_bitstrings(n_shots=20)
#     print("Samples:", bitstrings)
#     counts = {}
#     for b in bitstrings:
#         counts[b] = counts.get(b, 0) + 1
#     print("Counts:", counts)