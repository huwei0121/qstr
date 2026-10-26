"""Fresh randomized-circuit ORBIT feedback and strict call accounting."""

import time

import numpy as np

from .model import sample_circuits


class OrbitOracle:
    def __init__(
        self,
        model,
        budget,
        seed,
        mode="probability",
        shots=1000,
        circuit_policy="fresh",
        record_components=False,
    ):
        self.model, self.budget, self.mode, self.shots = model, budget, mode, shots
        if mode not in ("probability", "counts") or shots < 1:
            raise ValueError("Invalid feedback settings")
        self.sequence_rng = np.random.default_rng(np.random.SeedSequence([19431, int(seed)]))
        self.shot_rng = np.random.default_rng(np.random.SeedSequence([84241, int(seed)]))
        self.x, self.loss, self.failures, self.seconds = [], [], [], []
        self.log_floor_events = 0
        self.record_components = bool(record_components)
        self.components = []
        if circuit_policy not in ("fresh", "fixed"):
            raise ValueError("Invalid circuit policy")
        self.circuit_policy = circuit_policy
        self.fixed_circuits = None
        self.start = time.perf_counter()

    def __call__(self, x):
        if len(self.loss) >= self.budget:
            raise RuntimeError("Objective evaluation limit reached")
        x = np.asarray(x, float)
        if np.any(x < self.model.lower - 1e-12) or np.any(x > self.model.upper + 1e-12):
            raise ValueError("Optimizer proposed an out-of-bounds point")
        c = self.model.config
        if self.circuit_policy == "fixed" and self.fixed_circuits is not None:
            circuits = self.fixed_circuits
        else:
            circuits = sample_circuits(
                self.sequence_rng, c.circuits_per_evaluation, c.circuit_length, c.target_state
            )
            if self.circuit_policy == "fixed":
                self.fixed_circuits = circuits
        probabilities = self.model.probabilities(x, circuits)
        if self.mode == "counts":
            counts = self.shot_rng.binomial(self.shots, 1 - probabilities)
            failures = int(np.sum(counts))
            # Explicit finite-count extension, not the exact probability oracle.
            failure = (failures + 0.5) / (self.shots * c.circuits_per_evaluation + 1)
        else:
            failures = -1
            failure = float(np.mean(1 - probabilities))
            self.log_floor_events += int(failure < 1e-15)
            failure = max(failure, 1e-15)
        loss = float(np.log(failure))
        if self.record_components:
            # These are measured counts (or permitted per-circuit probability
            # feedback), never gate truth. Recording consumes no extra RNG.
            self.components.append((counts if self.mode == "counts" else 1 - probabilities).copy())
        self.x.append(x.copy())
        self.loss.append(loss)
        self.failures.append(failures)
        self.seconds.append(time.perf_counter() - self.start)
        return loss
