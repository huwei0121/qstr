"""Three-level, full-carrier Duffing dynamics and physical Clifford recovery.

Units are ns and radians/ns. No rotating-wave approximation, postselection,
SPAM, dissipation, pulse filtering, or ideal recovery is silently introduced.
The rotating frame is at the drive frequency and resets for each primitive.
The fixed-step eighth-order tableau is SciPy 1.11.4's DOP853 tableau.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.dop853_coefficients import A, B, C
from scipy.linalg import expm

RK_A = np.ascontiguousarray(A[:12, :12])
RK_B = np.ascontiguousarray(B)
RK_C = np.ascontiguousarray(C[:12])
SQRT2 = np.sqrt(2.0)
PHASES = np.array([0.0, np.pi, -np.pi / 2, np.pi / 2])
N_OP = np.diag([0.0, 1.0, 2.0])
A_OP = np.diag(np.sqrt([1.0, 2.0]), 1).astype(complex)
X2 = np.array([[0, 1], [1, 0]], complex)
Y2 = np.array([[0, -1j], [1j, 0]], complex)
IDEAL = np.array([expm(-1j * a * np.pi / 4) for a in (X2, -X2, Y2, -Y2)])

# The ordered decompositions are the public C3 qt_utils.py table at 48b7917.
# Index 0 is a *physical* two-pulse identity, not an ideal zero-duration gate.
WORDS = [
    (0, 1),
    (2, 0),
    (1, 3),
    (2, 0, 0),
    (1,),
    (0, 3, 1),
    (0, 0),
    (3, 1),
    (0, 3),
    (3,),
    (0,),
    (0, 2, 0),
    (2, 2),
    (3, 0),
    (0, 2),
    (3, 0, 0),
    (0, 2, 2),
    (0, 3, 0),
    (0, 0, 2, 2),
    (2, 1),
    (1, 2),
    (2,),
    (1, 2, 2),
    (0, 2, 1),
]


def compile_cliffords(primitives):
    out = []
    for word in WORDS:
        u = np.eye(primitives.shape[-1], dtype=complex)
        for p in word:
            u = primitives[p] @ u
        out.append(u)
    return np.array(out)


IDEAL_CLIFFORDS = compile_cliffords(IDEAL)


def group_index(u):
    overlaps = np.abs(np.einsum("cij,ij->c", IDEAL_CLIFFORDS.conj(), u)) / 2
    index = int(np.argmax(overlaps))
    if overlaps[index] < 1 - 1e-12:
        raise ValueError("Not a Clifford up to global phase")
    return index


MULTIPLY = np.array([[group_index(a @ b) for b in IDEAL_CLIFFORDS] for a in IDEAL_CLIFFORDS])
INVERSE = np.array([group_index(u.conj().T) for u in IDEAL_CLIFFORDS])


@dataclass(frozen=True)
class ModelConfig:
    qubit_frequency_ghz: float = 4.8
    anharmonicity_ghz: float = -0.2
    duration_ns: float = 10.0
    gaussian_sigma_ns: float = 10.0 / 6
    pwc_bins: int = 41
    substeps_per_bin: int = 40
    circuit_length: int = 60
    circuits_per_evaluation: int = 40
    target_state: int = 0


@njit(cache=True, nogil=True)
def _propagate(params, pwc, drive, wq, alpha, duration, width, bins, substeps):
    """Integrate two phases; opposite phases follow exact parity conjugation."""
    u = np.zeros((2, 3, 3), np.complex128)
    for p in range(2):
        for j in range(3):
            u[p, j, j] = 1.0
    h = duration / (bins * substeps)
    k = np.zeros((12, 2, 3, 3), np.complex128)
    v = np.empty_like(u)
    drift1 = wq - drive
    drift2 = 2 * drift1 + alpha
    for b in range(bins):
        for step in range(substeps):
            t0 = (b * substeps + step) * h
            for stage in range(12):
                t = t0 + RK_C[stage] * h
                if pwc:
                    iv, qv = params[b], params[bins + b]
                else:
                    z = (t - duration / 2) / width
                    g = np.exp(-0.5 * z * z)
                    iv = params[0] * g
                    qv = params[0] * params[1] * (-z / width) * g
                co, si = np.cos(drive * t), np.sin(drive * t)
                phase_factor = complex(co, -si)
                for p in range(2):
                    carrier = iv * co + qv * si if p == 0 else iv * si - qv * co
                    coupling = carrier * phase_factor
                    for r in range(3):
                        for j in range(3):
                            val = u[p, r, j]
                            for previous in range(stage):
                                val += h * RK_A[stage, previous] * k[previous, p, r, j]
                            v[p, r, j] = val
                    for j in range(3):
                        k[stage, p, 0, j] = -1j * coupling * v[p, 1, j]
                        k[stage, p, 1, j] = -1j * (
                            coupling.conjugate() * v[p, 0, j]
                            + drift1 * v[p, 1, j]
                            + SQRT2 * coupling * v[p, 2, j]
                        )
                        k[stage, p, 2, j] = -1j * (
                            SQRT2 * coupling.conjugate() * v[p, 1, j] + drift2 * v[p, 2, j]
                        )
            for p in range(2):
                for r in range(3):
                    for j in range(3):
                        for stage in range(12):
                            u[p, r, j] += h * RK_B[stage] * k[stage, p, r, j]
    out = np.empty((4, 3, 3), np.complex128)
    for p in range(2):
        for r in range(3):
            for j in range(3):
                out[2 * p, r, j] = u[p, r, j]
                out[2 * p + 1, r, j] = (-1.0) ** (r + j) * u[p, r, j]
    return out


def sample_circuits(rng, count, length, target=0):
    if length < 2 or target not in (0, 1):
        raise ValueError("Invalid circuit length or target")
    indices = rng.integers(24, size=(count, length - 1))
    product = np.zeros(count, dtype=int)
    for i in range(length - 1):
        product = MULTIPLY[indices[:, i], product]
    recovery = INVERSE[product]
    if target == 1:
        recovery = MULTIPLY[group_index(IDEAL[0] @ IDEAL[0]), recovery]
    return np.column_stack([indices, recovery])


def circuit_probabilities(cliffords, circuits, target=0):
    states = np.zeros((len(circuits), cliffords.shape[-1]), complex)
    states[:, 0] = 1.0
    for j in range(circuits.shape[1]):
        states = np.einsum("nij,nj->ni", cliffords[circuits[:, j]], states)
    probabilities = np.abs(states[:, target]) ** 2
    if np.any(probabilities > 1 + 2e-7) or not np.isfinite(probabilities).all():
        raise ArithmeticError("Nonphysical probability: refine propagation")
    return np.clip(probabilities, 0.0, 1.0)


class PulseModel:
    def __init__(self, config, pulse, reference, scale=None):
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        self.pulse = pulse
        if pulse not in ("drag", "pwc"):
            raise ValueError(pulse)
        self.reference = np.asarray(reference, float)
        self.dimension = 3 if pulse == "drag" else 2 * self.config.pwc_bins
        if self.reference.shape != (self.dimension + (pulse == "pwc"),):
            raise ValueError("PWC reference appends a fixed drive frequency")
        if scale is None:
            if pulse == "drag":
                scale = np.maximum(np.abs(self.reference), [1e-3, 0.1, 1e-3])
            else:
                b = self.config.pwc_bins
                scale = np.repeat(
                    [np.max(np.abs(self.reference[:b])), np.max(np.abs(self.reference[b : 2 * b]))],
                    b,
                )
                scale = np.maximum(scale, 1e-3)
        self.scale = np.asarray(scale, float)
        # Local calibration boxes: relative amplitude/DRAG +/-50/100%,
        # physical drive frequency +/-10%; PWC +/-50% of each channel peak.
        self.lower = (
            np.array([-0.5, -1.0, -0.1]) if pulse == "drag" else np.full(self.dimension, -0.5)
        )
        self.upper = -self.lower

    def physical(self, x):
        x = np.asarray(x, float)
        if x.shape != (self.dimension,) or not np.isfinite(x).all():
            raise ValueError("Invalid parameter vector")
        values = self.reference.copy()
        values[: self.dimension] += self.scale * x
        return values

    def initial(self, seed):
        rng = np.random.default_rng(np.random.SeedSequence([7341, int(seed)]))
        # Literal +/-5% per physical parameter; PWC carrier remains fixed.
        perturbed = self.reference[: self.dimension] * (
            0.05 * rng.choice([-1.0, 1.0], self.dimension)
        )
        return perturbed / self.scale

    @lru_cache(maxsize=16)
    def _primitive(self, x):
        p = self.physical(x)
        c = self.config
        result = _propagate(
            p,
            self.pulse == "pwc",
            p[-1],
            2 * np.pi * c.qubit_frequency_ghz,
            2 * np.pi * c.anharmonicity_ghz,
            c.duration_ns,
            c.gaussian_sigma_ns,
            c.pwc_bins,
            c.substeps_per_bin,
        )
        residual = np.max(np.abs(result.conj().transpose(0, 2, 1) @ result - np.eye(3)))
        if residual > 1e-8:
            raise ArithmeticError(f"Unitarity residual {residual}: refine propagation")
        return result

    def primitives(self, x):
        return self._primitive(tuple(np.asarray(x, float)))

    def probabilities(self, x, circuits):
        return circuit_probabilities(
            compile_cliffords(self.primitives(x)), circuits, self.config.target_state
        )

    def diagnostics(self, x):
        matrices = self.primitives(x)
        block = matrices[:, :2, :2]
        weight = np.sum(np.abs(block) ** 2, axis=(1, 2))
        overlap = np.abs(np.einsum("pij,pij->p", IDEAL.conj(), block)) ** 2
        error = 1 - (weight + overlap) / 6
        return dict(
            mean_gate_infidelity=float(error.mean()),
            gate_infidelities=error.tolist(),
            mean_leakage=float((1 - weight / 2).mean()),
            unitarity_residual=float(
                np.max(np.abs(matrices.conj().transpose(0, 2, 1) @ matrices - np.eye(3)))
            ),
        )

    def independent_primitives(self, x):
        """Adaptive SciPy DOP853; integrate laboratory H then change frame.

        PWC discontinuities are exact integration boundaries. This also tests
        rotating-frame signs, phase conventions and the custom RK loop.
        """
        p, c = self.physical(x), self.config
        wq, alpha, wd = 2 * np.pi * c.qubit_frequency_ghz, 2 * np.pi * c.anharmonicity_ghz, p[-1]
        drift = wq * N_OP + alpha / 2 * (N_OP @ N_OP - N_OP)
        results = []
        edges = (
            np.linspace(0, c.duration_ns, c.pwc_bins + 1)
            if self.pulse == "pwc"
            else np.array([0.0, c.duration_ns])
        )
        for phase in PHASES:
            u = np.eye(3, dtype=complex)
            for b, (t0, t1) in enumerate(zip(edges[:-1], edges[1:])):

                def rhs(t, y):
                    if self.pulse == "pwc":
                        iv, qv = p[b], p[c.pwc_bins + b]
                    else:
                        z = (t - c.duration_ns / 2) / c.gaussian_sigma_ns
                        g = np.exp(-z * z / 2)
                        iv, qv = p[0] * g, -p[0] * p[1] * z * g / c.gaussian_sigma_ns
                    field = iv * np.cos(wd * t + phase) + qv * np.sin(wd * t + phase)
                    return (
                        -1j * (drift + field * (A_OP + A_OP.conj().T)) @ y.reshape(3, 3)
                    ).ravel()

                sol = solve_ivp(rhs, (t0, t1), u.ravel(), method="DOP853", rtol=2e-12, atol=2e-14)
                if not sol.success:
                    raise ArithmeticError(sol.message)
                u = sol.y[:, -1].reshape(3, 3)
            results.append(np.diag(np.exp(1j * wd * c.duration_ns * np.arange(3))) @ u)
        return np.array(results)

    def serialize(self):
        return dict(
            config=asdict(self.config),
            pulse=self.pulse,
            reference=self.reference.tolist(),
            scale=self.scale.tolist(),
        )
