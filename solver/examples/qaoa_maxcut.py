"""Sample a five-qubit QAOA objective and optimize its ten circuit angles.

The graph is K(3, 2), with five alternating layers and initial state |+>**5.
Parameters are beta[0:5], gamma[0:5]. Each layer applies exp(-i*gamma*sum(ZZ))
followed by exp(-i*beta*sum(X)). Measurements minimize the negative cut mean.
This small NumPy example uses multinomial sampling; a hardware objective can
replace the callable without changing the optimization interface.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from qstr import minimize_with_preset


class MaxCutObjective:
    """A sampled scalar objective with an independent measurement stream."""

    def __init__(self, seed: int = 17, shots: int = 1000) -> None:
        if shots < 1:
            raise ValueError("shots must be positive")
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        states = np.arange(32)
        edges = [(left, right) for left in range(3) for right in (3, 4)]
        self.cut_values = sum(
            ((states >> left) & 1) != ((states >> right) & 1) for left, right in edges
        ).astype(float)
        self.zz_values = 6 - 2 * self.cut_values
        self.bit_pairs = [
            (states[(states & (1 << qubit)) == 0], states[(states & (1 << qubit)) != 0])
            for qubit in range(5)
        ]

    def probabilities(self, angles: np.ndarray) -> np.ndarray:
        """Propagate a circuit; the optimizer receives only the sampled mean."""
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (10,) or not np.all(np.isfinite(angles)):
            raise ValueError("angles must be a finite vector of length ten")
        state = np.full(32, 1 / np.sqrt(32), dtype=complex)
        for beta, gamma in zip(angles[:5], angles[5:]):
            state *= np.exp(-1j * gamma * self.zz_values)
            cosine, sine = np.cos(beta), -1j * np.sin(beta)
            for left, right in self.bit_pairs:
                left_values, right_values = state[left].copy(), state[right].copy()
                state[left] = cosine * left_values + sine * right_values
                state[right] = sine * left_values + cosine * right_values
        probabilities = np.abs(state) ** 2
        return probabilities / probabilities.sum()

    def __call__(self, angles: np.ndarray) -> float:
        counts = self.rng.multinomial(self.shots, self.probabilities(angles))
        self.calls += 1
        return -float(counts @ self.cut_values) / self.shots

    def expected_value(self, angles: np.ndarray) -> float:
        """Independent exact diagnostic, called only after optimization."""
        return -float(self.probabilities(angles) @ self.cut_values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evaluations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    objective = MaxCutObjective(seed=arguments.seed + 10)
    result = minimize_with_preset(
        objective,
        preset="qaoa_maxcut",
        max_evaluations=arguments.max_evaluations,
        seed=arguments.seed,
    )
    summary = {
        "evaluations": result.nfev,
        "shots": objective.calls * objective.shots,
        "observed_value": result.fun,
        "independent_expected_value": objective.expected_value(result.x),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
