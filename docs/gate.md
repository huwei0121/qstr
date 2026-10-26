# Two-qutrit gate calibration

This benchmark calibrates four single-qubit rotations on a coupled two-qubit
device. It implements the model and sampling protocol in Wittler et al.,
*Integrated Tool Set for Control, Calibration, and Characterization of Quantum
Devices*, Physical Review Applied 15, 034080 (2021),
[DOI](https://doi.org/10.1103/PhysRevApplied.15.034080), with components from
[C3](https://github.com/q-optimize/c3/tree/48b7917eb8d84e1471ff9fb911129c354e752dbf).
The original publication's exact calibrated pulse and random seeds were not
available in the located public materials. The supplied initial point and
seeds define the reproducible comparison in this package.

## Physical variables and normalization

The optimizer sees three dimensionless coordinates, in this exact order:

| Index | Control | Physical range | Conversion from x |
|---|---|---|---|
| 0 | Gaussian envelope amplitude A | 0.2 to 0.6 V | A = 0.4 + 0.2 x[0] V |
| 1 | DRAG coefficient delta | -5 to 3, dimensionless | delta = -1 + 4 x[1] |
| 2 | Envelope frequency offset f | -60 to -40 MHz | f = -50 + 10 x[2] MHz |

All three normalized coordinates lie in [-1, 1]. In general,
physical = lower + (upper - lower)(x + 1)/2. Frequencies in the C3 Quantity
with unit "Hz 2pi" are converted internally to angular frequencies; the table
reports ordinary frequencies. The frequency offset is relative to the local
oscillator, not relative to zero frequency. For target qubit 0 the local
oscillator is 5.05 GHz. Do not apply another 2 pi factor to the stored
VoltsToHertz conversion coefficient.

The initial normalized coordinates are
[-0.054559463003214105, 0.08264154056073997, -0.11956622745212653]. They correspond
to amplitude 0.3890881073993572 V, delta -0.6694338377570401, and offset
-51.19566227452127 MHz. The initial point is a fixed coarse calibration from
the reconstruction; regenerating it is unnecessary for reproducing the
reported optimizer comparison.

The target qubit's +X, +Y, -X and -Y pi/2 gates share these three controls.
The C3 EnvelopeDrag convention uses an imaginary component equal to minus
delta times the derivative of the Gaussian envelope with respect to AWG bins.
Thus its dimensionless delta must not be substituted directly for a DRAG
coefficient from a different pulse convention.

Each qubit retains three levels. Their coupled Hilbert space has dimension 9.
The qubit frequencies are 5 and 5.6 GHz; anharmonicities are -210 and -240 MHz;
coupling is 20 MHz. Relaxation times are 27 and 23 microseconds, and pure
dephasing times are 39 and 31 microseconds. The initial state has temperature
50 mK. Pulses last 7 ns; the AWG rate is 2 GHz; the response rise time is 0.3 ns.
Readout confusion probabilities come from the upstream two-qubit notebook.

## Model construction and numerical propagation

The vendored C3 source is unchanged. Its selected notebook setup cells have
been extracted into the ordinary function gate_support.device_setup.build_setup.
The local model applies four explicit reconstruction choices:

1. Enable Lindblad evolution, the thermal initial state and readout confusion.
2. Use separate relaxation and pure-dephasing collapse channels. The upstream
   library adds their operators before constructing a channel; the local
   PaperModel subclass implements the sum of independent Lindblad channels
   required by the physical model.
3. Add the upstream Response component to both control chains.
4. Widen the notebook's frequency-offset range to [-60, -40] MHz, matching the
   public single-qubit example and covering both dressed resonances.

Production propagation uses an 800 GHz signal grid and symmetric Strang
splitting with 0.05 ns dissipative blocks. Hamiltonian evolution uses exact
9 by 9 exponentials on the piecewise-constant signal grid. The negative
rotation channels follow from parity conjugation of the corresponding
positive rotations; a commutator check verifies applicability of this
symmetry. The final virtual frame correction is the upstream convention.
Both optimization methods use the same propagation and sampling code.

## Feedback, randomness and cost

Every optimizer evaluation generates 25 new length-100 Clifford circuits,
including the recovery operation, and takes 1000 independent Bernoulli shots
per circuit. Feedback is the mean frequency of reporting the target qubit
as excited. One evaluation therefore costs 25,000 simulated measurement shots.
An ORBIT failure probability of 0.23 is a circuit-level readout quantity;
it is not a single-gate infidelity of 23 percent.

For evaluation k starting at one, the circuit seed is derived by NumPy
SeedSequence([seed, k, 11]). Measurement counts use an independent
SeedSequence([seed, k, 21]). The raw output stores circuit lists, shot seed
paths, individual counts and physical costs in sampling_raw.jsonl.gz.
Neither exact probabilities nor validation results are returned to the
optimizer.

After search, native recommendations are frozen at the configured checkpoints.
They are then evaluated on a common independent set of 4096 circuits, with
seed 9431057, using exact probabilities. The reported standard error measures
variation across this finite validation ensemble. Validation is offline and
uses no simulated shots; its number of circuit evaluations is recorded
separately. For software smoke tests, validation.circuits may be reduced;
such outputs must retain that setting and must not be merged into the full
comparison.

## Algorithms and returned points

Q-STR calls qstr.minimize_with_preset with preset gate_calibration. The
100-evaluation default comprises local stages capped at 50 evaluations,
initial radius 0.08 and final radius 0.0001. Its recommendation is the point
with the smallest completed observation-group mean.

CMA-ES calls the unmodified C3 cmaes wrapper, backed by cma 3.3.0, with
population 25, spread 0.1 and init_point=False. Proposals are unbounded in
latent coordinates. C3's Quantity reflection maps each scalar z to
2 arccos(cos((z + 1) pi/2))/pi - 1 before physical evaluation. This is distinct
from using the CMA bounds option.

A strict evaluation cap can interrupt a population. All evaluated points
are charged, but the incomplete population is not passed to tell. The
reported CMA recommendation remains xbest from completed populations.
Accordingly, a CMA run needs at least 25 evaluations. At earlier checkpoints
there is no native recommendation; the results contain no fabricated point.
Checkpoint replay consumes only already paid feedback and verifies that its
coordinates match the recorded prefix to 1e-12.

## Installation and validation

Install using the reproduction package's gate environment. The pinned runtime
is Python 3.10, TensorFlow 2.15.0, tensorflow-probability 0.23.0, NumPy 1.26.4,
SciPy 1.11.4, cma 3.3.0 and COBYQA 1.1.4. No Qiskit installation is required
for this backend. Source, dependency pins and the Apache-2.0 license are
included locally; installation still downloads Python dependencies from PyPI.

Migration validation is saved in validation/gate_equivalence.json:
two parameter points produce identical probabilities on 12 circuits each;
50 circuit samples produce identical counts and 50,000-shot accounting;
four CMA budgets reproduce every coordinate, feedback value, native return
and partial population. The summary includes both algorithm smoke runs and
a full-budget Q-STR run with 4096 validation circuits. All 98 paid observations,
the final return and every checkpoint quality in that full-budget run match
the stored confirmation run exactly. These checks test the implementation
migration; they do not introduce a new optimizer-performance claim.
