"""Variable-circuit sampling for the established paper C3 two-qutrit model.

One observation is a failure frequency from one RB circuit and 1000 shots.
Paired sampling shares circuits only; measurement draws use independent streams.
Neither public sample statistics nor history expose latent probabilities.
"""

import gzip
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .model import SplitPropagator, build, circuit_probabilities, circuits

DEFAULT_X0 = np.array([-0.054559463003214105, 0.08264154056073997, -0.11956622745212653])


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_point(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all() or np.any(np.abs(x) > 1 + 1e-12):
        raise ValueError("C3 point must lie in the native three-parameter box")
    return x.copy()


def statistics(n, mean, m2):
    """Observed mean, variance of that mean, derivative of identity link."""
    if int(n) != n or n < 2 or not np.isfinite([mean, m2]).all() or m2 < -1e-14:
        raise ValueError("At least two finite circuit observations are required")
    return float(mean), max(float(m2), 0.0) / (int(n) * (int(n) - 1)), 1.0


class GateDevice:
    """Existing physical model and equivalent, previously checked propagation."""

    def __init__(self, target=0, output=None):
        self.target = int(target)
        self.ns = build(self.target, 800e9, output)
        self.prop = SplitPropagator(self.ns, 0.05, parity=True, tensor_action=True)
        self._operators = OrderedDict()

    def make_circuits(self, n, seed):
        return circuits(self.ns, int(n), 100, int(seed))

    def probabilities(self, x, sequences):
        key = tuple(check_point(x))
        if key not in self._operators:
            operators = self.prop(x)
            canonical = np.asarray(self.ns["parameter_map"].get_parameters_scaled()).reshape(-1)
            if not np.allclose(x, canonical, rtol=0, atol=1e-12):
                raise RuntimeError("Parameter map changed a native-box point")
            self._operators[key] = operators
            if len(self._operators) > 16:
                self._operators.popitem(last=False)
        else:
            self._operators.move_to_end(key)
        p, diag = circuit_probabilities(self.ns, self._operators[key], sequences)
        if (
            not np.isfinite(p).all()
            or p.min() < -1e-8
            or p.max() > 1 + 1e-8
            or diag["max_trace_error"] > 1e-7
        ):
            raise RuntimeError("Invalid C3 physical probabilities")
        return np.clip(p, 0.0, 1.0), diag


class GateOracle:
    base_samples = 25
    unit_cost = 1000
    min_samples = 2
    statistics = staticmethod(statistics)

    def __init__(self, budget, seed, output=None, target=0, x0=None, device=None):
        if int(budget) != budget or budget < 0 or int(seed) != seed or seed < 0:
            raise ValueError("Budget and seed must be nonnegative integers")
        self.budget, self.seed, self.spent = int(budget), int(seed), 0
        self.x0 = check_point(DEFAULT_X0 if x0 is None else x0)
        self.lower, self.upper = -np.ones(3), np.ones(3)
        self.history = []
        self._batch_count = 0
        self.output = None if output is None else Path(output)
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
        self._device = device if device is not None else GateDevice(target, self.output)
        self._raw = None
        self._in_memory_raw = []
        if self.output is not None:
            path = self.output / "sampling_raw.jsonl.gz"
            if path.exists():
                raise FileExistsError(path)
            self._raw = gzip.open(path, "wt", encoding="utf-8")

    def _check_cost(self, n, endpoints):
        if int(n) != n or n < self.min_samples:
            raise ValueError("C3 sampling requires an integer n >= 2")
        cost = int(n) * int(endpoints) * self.unit_cost
        if self.spent + cost > self.budget:
            raise ValueError("Strict physical-shot budget would be exceeded")
        return int(n), cost

    def _new_batch(self):
        self._batch_count += 1
        index = self._batch_count
        # Domain-separated SeedSequences: shared circuit stream, independent
        # left/right measurement streams. Each batch receives new circuits.
        paths = {
            name: [self.seed, index, domain]
            for name, domain in [("circuits", 11), ("left_shots", 21), ("right_shots", 22)]
        }
        circuit_seed = int(np.random.SeedSequence(paths["circuits"]).generate_state(1)[0])
        return f"c3_s{self.seed}_b{index}", circuit_seed, paths

    def _endpoint(self, x, sequences, seed_path, batch_id, role):
        p, diag = self._device.probabilities(x, sequences)
        counts = np.random.default_rng(np.random.SeedSequence(seed_path)).binomial(
            self.unit_cost, p
        )
        observations = counts.astype(float) / self.unit_cost
        mean = float(observations.mean())
        centered = observations - mean
        result = dict(
            n=len(observations),
            mean=mean,
            m2=float(centered @ centered),
            cost=len(observations) * self.unit_cost,
            batch_id=batch_id,
        )
        row = dict(
            **result,
            x=x.tolist(),
            role=str(role),
            counts=counts.tolist(),
            shot_seed_path=seed_path,
            shots_per_circuit=self.unit_cost,
            diagnostics=diag,
        )
        return result, row, observations

    def _record(self, raw):
        encoded = json.dumps(plain(raw), allow_nan=False)
        if self._raw is not None:
            self._raw.write(encoded + "\n")
            self._raw.flush()
        else:
            self._in_memory_raw.append(raw)

    def sample(self, x, n, role="sample"):
        x = check_point(x)
        n, cost = self._check_cost(n, 1)
        t = time.perf_counter()
        batch_id, circuit_seed, paths = self._new_batch()
        seq = self._device.make_circuits(n, circuit_seed)
        result, row, _ = self._endpoint(x, seq, paths["left_shots"], batch_id, role)
        self.spent += cost
        row.update(
            cost_end=self.spent,
            circuit_seed=circuit_seed,
            seconds=time.perf_counter() - t,
            kind="single",
        )
        self.history.append(row)
        self._record(dict(**row, sequences=seq, circuit_seed_path=paths["circuits"]))
        return result

    def sample_pair(self, x, y, n, role="pair"):
        x, y = check_point(x), check_point(y)
        n, cost = self._check_cost(n, 2)
        t = time.perf_counter()
        batch_id, circuit_seed, paths = self._new_batch()
        seq = self._device.make_circuits(n, circuit_seed)
        left, lr, a = self._endpoint(x, seq, paths["left_shots"], batch_id + ":left", role)
        right, rr, b = self._endpoint(y, seq, paths["right_shots"], batch_id + ":right", role)
        cross_m2 = float((a - a.mean()) @ (b - b.mean()))
        elapsed = time.perf_counter() - t
        for side, row in [("left", lr), ("right", rr)]:
            self.spent += row["cost"]
            row.update(
                cost_end=self.spent,
                circuit_seed=circuit_seed,
                kind="pair",
                pair_id=batch_id,
                pair_side=side,
                seconds=elapsed / 2,
                cross_m2=cross_m2,
            )
            self.history.append(row)
        self._record(
            dict(
                kind="pair",
                batch_id=batch_id,
                left=lr,
                right=rr,
                cross_m2=cross_m2,
                cost=cost,
                cost_end=self.spent,
                circuit_seed=circuit_seed,
                circuit_seed_path=paths["circuits"],
                sequences=seq,
                seconds=elapsed,
                coupling="shared circuits; independent endpoint shots",
            )
        )
        return dict(left=left, right=right, cross_m2=cross_m2, cost=cost, batch_id=batch_id)

    def holdout(self, points, n=512, seed=991040):
        """Call only after search; exact probabilities are offline diagnostics."""
        seq = self._device.make_circuits(n, seed)
        rows = []
        for name, point in points.items():
            start = time.perf_counter()
            p, diag = self._device.probabilities(check_point(point), seq)
            rows.append(
                dict(
                    name=name,
                    x=plain(point),
                    mean=float(p.mean()),
                    circuit_se=float(p.std(ddof=1) / np.sqrt(n)),
                    probabilities=p.tolist(),
                    diagnostics=diag,
                    seconds=time.perf_counter() - start,
                )
            )
        return dict(
            circuits=n, seed=seed, sequences=seq, values=rows, oracle_queries=len(rows), shots=0
        )

    def close(self):
        if self._raw is not None:
            self._raw.close()
            self._raw = None
