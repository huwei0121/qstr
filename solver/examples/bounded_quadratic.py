"""Minimize a bounded scalar objective with complete evaluation accounting."""

import json

import numpy as np

from qstr import SolverOptions, minimize


def main() -> None:
    result = minimize(
        lambda x: float(np.sum((x - 0.2) ** 2)),
        np.zeros(4),
        max_evaluations=120,
        bounds=(-np.ones(4), np.ones(4)),
        seed=7,
        options=SolverOptions(initial_radius=0.1),
    )
    print(json.dumps({"x": result.x.tolist(), "fun": result.fun, "nfev": result.nfev}))


if __name__ == "__main__":
    main()
