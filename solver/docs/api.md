# Python API

Q-STR minimizes a scalar objective using quadratic interpolation models,
optional structured coordinates, and budget-aware restarts. The objective
receives parameter vectors in the original problem coordinates.

## `minimize`

```python
from qstr import minimize

result = minimize(
    objective,
    x0,
    max_evaluations,
    bounds=None,
    basis_factory=None,
    seed=0,
    options=None,
)
```

The arguments after `max_evaluations` are keyword-only.

| Argument | Meaning |
| --- | --- |
| `objective` | Callable accepting a finite one-dimensional NumPy array and returning a finite scalar to minimize. Each invocation receives its own copy of the parameter vector. |
| `x0` | Nonempty finite initial vector of length `n`. It must satisfy the supplied bounds. |
| `max_evaluations` | Positive integer limiting the total number of objective calls, including initialization, repetitions, and restarts. |
| `bounds` | `None`, or `(lower, upper)` with two finite vectors of shape `(n,)` satisfying `lower < upper`. |
| `basis_factory` | `None`, or `basis_factory(stage_index, stage_seed)` returning an orthonormal basis matrix or `None`. See [structured coordinates](#structured-coordinates). |
| `seed` | Integer used for the search-coordinate schedule. The default is `0`. |
| `options` | `None`, a `SolverOptions` instance, or a mapping of option names to values. A mapping overrides the generic defaults below. Unknown option names raise `ValueError`. |

For example:

```python
import json
import numpy as np
from qstr import SolverOptions, minimize

def objective(x):
    return float(np.sum((x - 0.25) ** 2))

result = minimize(
    objective,
    np.zeros(4),
    max_evaluations=120,
    bounds=(-np.ones(4), np.ones(4)),
    options=SolverOptions(initial_radius=0.2, stage_evaluations=40),
)

print(result.x, result.fun, result.nfev)
points = result.history.x       # shape (nfev, 4)
values = result.history.fun     # shape (nfev,)
serialized = json.dumps(result.to_dict())
```

### Bounds and structured coordinates

Bounds are a pair of lower and upper vectors. For example, a three-dimensional
box uses `bounds=(-np.ones(3), np.ones(3))`.

For stage index `j`, starting at zero, the basis factory receives
`stage_seed = seed + 173 + 1009*j`. Its result must have shape `(n, d)` with
`1 <= d <= n`, finite entries, and orthonormal columns. A `None` result or an
exact identity matrix selects native coordinates. A nonidentity square matrix
performs a full-dimensional coordinate transformation.

In an unbounded structured stage, the model coordinates `z` map to
`x = center + basis @ z`. With bounds, an anchored hyperbolic-tangent map keeps
proposals strictly inside the box and has derivative `basis` at `z = 0`.
Such stages require an interior center. Native-coordinate stages also permit
centers on the boundary.

The factory is called once per attempted stage, including an attempted restart
that subsequently stops because of insufficient remaining budget. It is not
called speculatively to estimate future dimensions.

```python
def basis_factory(stage_index, stage_seed):
    rng = np.random.default_rng(stage_seed)
    basis, _ = np.linalg.qr(rng.normal(size=(20, 5)))
    return basis
```

### Randomness and evaluation accounting

`seed` controls basis generation through the factory. The objective owns its
measurement random stream. Create and seed that stream explicitly when
reproducible noisy evaluations are needed:

```python
measurement_rng = np.random.default_rng(17)

def noisy_objective(x):
    exact_value = np.sum((x - 0.25) ** 2)
    return float(exact_value + measurement_rng.normal(0.0, 0.001))
```

With `repetitions=r`, up to `r` consecutive calls at the same point form one
observation group. Its arithmetic mean is passed to the local model. Individual
calls all count toward `max_evaluations`; the final group in a stage may be
shorter to respect its budget. A subsequent visit to the same point forms a
separate group. The result selects the smallest observed group mean across all
stages, retaining the first group in a tie.

An objective call may itself execute many circuits, shots, or propagations.
The caller determines and records that physical cost. Q-STR counts calls to
the supplied objective and stores their individual feedback values.

## `SolverOptions`

```python
from qstr import SolverOptions

options = SolverOptions(initial_radius=0.1, repetitions=3)
same_options = SolverOptions.from_mapping(options.to_dict())
```

`SolverOptions` is an immutable dataclass. All defaults below describe the
generic solver. Problem presets supply their own numerical settings.

### Local models and radii

| Option | Default | Meaning |
| --- | --- | --- |
| `initial_radius` | `0.1` | Initial trust-region radius before progress scaling. Must be positive. |
| `final_radius` | `1e-6` | Target resolution of a local solve. Must satisfy `0 < final_radius <= initial_radius`. |
| `interpolation_factor` | `2.0` | Controls the interpolation-point count in dimension `d`: `p = max(d+1, min((d+1)*(d+2)//2, int(interpolation_factor*d+1)))`. Must be positive. |
| `minimum_radius` | `1e-5` | Lower bound on each stage's initial radius after progress scaling. Must be positive. |
| `progress_exponent` | `0.0` | Coefficient used to scale the initial radius as the incumbent improves. Zero disables this scaling. |
| `progress_reference_stage` | `-1` | Negative values use the initial group mean as the progress reference. A nonnegative integer selects the incumbent value at the start of that stage, or at the current stage while that index has not yet been reached. |

Let `base_radius` be the initial radius selected for the current stage, `best`
the incumbent group mean, and `reference` the progress reference. The actual
initial radius is

```text
max(minimum_radius,
    base_radius * exp(clip(progress_exponent * (best - reference), -700, 0)))
```

The local solver's final resolution is
`min(final_radius, 0.2 * actual_initial_radius)`. Radii are measured in local
model coordinates. The final resolution controls a local solve's stopping
criterion; it does not specify an objective-error tolerance.

### Stages, stopping, and budget reserves

| Option | Default | Meaning |
| --- | --- | --- |
| `stage_evaluations` | `None` | Planned maximum individual objective calls for a stage. `None` uses the total evaluation budget as its planned cap. A supplied value must be a positive integer. |
| `max_stages` | `1000` | Maximum number of local solves. Positive integer. |
| `stagnation_patience` | `0` | Number of individual calls without a qualifying improvement before requesting a restart. Zero disables this criterion. |
| `min_stage_evaluations` | `0` | Minimum individual calls before the stagnation criterion can act. Nonnegative integer. |
| `improvement_tolerance` | `0.0` | Required absolute reduction from the stage's improvement reference. The reference is initialized by the first group and updated after each qualifying improvement. Nonnegative. |
| `restart_reserve_factor` | `2.0` | Nonnegative factor used in the minimum remaining budget required for a restart. |
| `min_restart_evaluations` | `0` | Additional absolute lower bound on the restart reserve. Nonnegative integer. |

For `p` interpolation points and `r` repetitions, the reserve is
`max(min_restart_evaluations, ceil(restart_reserve_factor*p*r))`. A stage is
extended to consume the remaining budget when its planned end would leave a
positive tail smaller than this reserve. After the initial stage, a restart
with less than its computed reserve is skipped. The initial group is evaluated
before the first stage and is included in the total budget.

Stagnation checks begin only after
`max(p*r + r, min_stage_evaluations)` calls in the current stage. A stagnation
restart additionally requires more than `p*r` calls remaining and at least
the restart reserve. Every restart rebuilds a local model around the point
with the lowest observed group mean.

### Repeated observations and warmup

| Option | Default | Meaning |
| --- | --- | --- |
| `repetitions` | `1` | Positive integer specifying individual calls per observation group, subject to the remaining budget. |
| `warmup_stages` | `0` | Number of initial stages using warmup overrides. Nonnegative integer. |
| `warmup_radius` | `None` | Positive initial-radius override for warmup stages. `None` uses `initial_radius`. |
| `warmup_evaluations` | `None` | Positive integer overriding the planned warmup stage budget. `None` uses `stage_evaluations`. |
| `warmup_interpolation_factor` | `None` | Positive interpolation-factor override for warmup stages. `None` uses `interpolation_factor`. |

## Results

`minimize` and `minimize_with_preset` return an `OptimizationResult`.

| Field | Meaning |
| --- | --- |
| `x`, `fun` | Recommended parameter vector and its observed group mean. |
| `nfev`, `max_evaluations` | Actual objective-call count and requested total limit. |
| `history` | `EvaluationHistory` with arrays `x`, `fun`, and `elapsed_seconds`, in objective-call order. |
| `observation_groups` | Tuple of `ObservationGroup` records: `end_evaluation`, `x`, `mean`, and `repetitions`. |
| `stages` | Tuple of `StageResult` records describing local solves. |
| `options` | The `SolverOptions` used for the run. A generic `stage_evaluations=None` remains `None` in this record; stage records give the realized limits. |
| `elapsed_seconds` | Total elapsed wall-clock time. |
| `seed` | Basis-schedule seed. |
| `restart_decisions` | Budget decisions extending an active stage or skipping an underfunded restart. |
| `metadata` | Dictionary for problem-specific reporting. The preset interface records the preset name and coordinate description here. |

Group endpoints and stage evaluation indices are one-based cumulative counts;
stage endpoints are inclusive. For example:

```python
group = result.observation_groups[0]
first = group.end_evaluation - group.repetitions
last = group.end_evaluation
assert group.mean == float(result.history.fun[first:last].mean())
```

Each `StageResult` includes its `index`, `start_evaluation`, `end_evaluation`,
`evaluations`, local `dimension`, `interpolation_points`,
`initialization_evaluations`, `initial_radius`, `final_radius`, `coordinate_map`,
`basis_seed`, `stop_reason`, `solver_message`, `solver_recommendation`,
`warnings`, `model_evaluations`, `requested_evaluation_limit`,
`evaluation_limit`, and `restart_reserve`.

`model_evaluations` counts averaged feedback calls. `evaluations` counts
individual objective calls. `initialization_evaluations` counts the first
`p*r` individual calls of the stage, capped by its actual length.
`solver_recommendation` records the local solver's final parameter vector
when available; the overall recommendation is selected from observation groups.

The stage `stop_reason` is one of `solver_termination`, `stagnation`,
`evaluation_budget`, or `fallback_evaluation`. COBYQA may itself return when
its evaluation limit is reached; in that case `stop_reason` is
`solver_termination` and `solver_message` describes its exit condition.
An empty stage is advanced by a paid center evaluation, recorded as
`fallback_evaluation`.

`result.to_dict()` returns independent nested records with NumPy arrays and
scalars converted to JSON-compatible lists and scalars. Caller-supplied
metadata should likewise contain JSON-compatible values for JSON export.

## Problem presets

```python
from qstr import PRESET_NAMES, get_preset, minimize_with_preset

configuration = get_preset("gate_calibration")
```

`get_preset(name)` returns an independent dictionary containing the dimension,
initial point, bounds, budget, numerical options, coordinate convention,
feedback definition, and nominal cost per objective call.

| Preset | Dimension | Default budget | Expected objective feedback |
| --- | --- | --- | --- |
| `pulse_calibration` | 82 | 1000 | Logarithm of mean failure probability over 40 random circuits. Requires an instance-specific initial point. |
| `gate_calibration` | 3 | 100 | Mean failure frequency from 25 circuits with 1000 shots each. |
| `qaoa_maxcut` | 10 | 550 | Negative mean cut value from 1000 shots. Angles are ordered by all beta layers followed by all gamma layers. |
| `ising_state_transfer` | 100 | 2000 | State-transfer infidelity from one noisy propagation. Input coordinates are normalized amplitudes; physical amplitudes equal `1000*x`. |

```python
result = minimize_with_preset(
    objective,
    preset="gate_calibration",
    x0=None,
    max_evaluations=None,
    seed=0,
    options={"initial_radius": 0.04},
)
```

Only `objective` is positional. `preset` is required. `x0` and
`max_evaluations` default to the preset values. The objective must accept the
documented problem coordinates and return the documented feedback scale.
The wrapper applies the preset's basis internally, so pass the objective in
original problem coordinates. A simulator or hardware interface is supplied
by the caller.

The two forms of `options` have different purposes:

- A mapping such as `{"initial_radius": 0.04}` overrides only those preset
  fields and retains its other settings.
- A `SolverOptions` instance replaces all preset numerical settings. Fields
  omitted from its constructor take the generic defaults, including
  `warmup_stages=0` and `repetitions=1`.

Preset bounds and basis families remain selected by the problem name; the
basis schedule receives the resulting `warmup_stages` setting. For other
coordinate conventions, bounds, or basis families, use `minimize` directly.
The preset's `evaluation_cost` describes its nominal measurement protocol.
The wrapper records paid objective calls and leaves hardware-cost accounting
to the caller.
