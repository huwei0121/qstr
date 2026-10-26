# Pulse calibration: Duffing dynamics and ORBIT

This benchmark implements the PWC82 problem family studied by K. Pack,
S. Machnes and F. K. Wilhelm in [Benchmarking Optimization Algorithms for
Automated Calibration of Quantum Devices](https://arxiv.org/abs/2509.08555).
It reproduces the complete experiment used in this project's comparison.
The paper's PWC optimizer drivers and complete hyperparameters were not
available in its data release; the recorded six baseline settings are an
explicit reconstruction. They are provided in configs/full.json.

## Physical model and coordinates

The simulated transmon has levels 0, 1 and 2. With number operator N and
annihilation operator a, the laboratory Hamiltonian is

    H(t) = omega_q N + alpha/2 (N^2 - N)
           + [I(t) cos(omega_d t + phi) + Q(t) sin(omega_d t + phi)] (a + a*) .

Frequencies are angular frequencies in radians/ns: omega_q = 2*pi*4.8 and
alpha = -2*pi*0.2. Each primitive lasts 10 ns. The drive frequency, 41
piecewise-constant I amplitudes, 41 Q amplitudes and their scaling constants
are included in pulse_support/model.json. The carrier frequency is fixed
during PWC optimization. For each amplitude j,

    physical_amplitude[j] = reference[j] + scale[j] * x[j],
    -0.5 <= x[j] <= 0.5.

The first 41 coordinates control I; the last 41 control Q. Initial points
perturb every reference amplitude by a seeded choice of plus or minus 5%.
The initial RNG uses SeedSequence([7341, seed]). There are 82 variables.

Propagation uses the full carrier, three levels and fixed-step eighth-order
DOP853 with 40 substeps per bin. An independent adaptive DOP853 propagation
is also available for model checks. Primitive phases are 0, pi, -pi/2 and
pi/2; the rotating frame resets for each primitive. Physical Clifford
decompositions implement all gates, including recovery. There is no ideal
recovery, rotating-wave approximation, SPAM, filtering or dissipation in
this particular benchmark.

## Paid observation and independent metric

Every objective call samples 40 fresh circuits of length 60 Clifford gates,
including recovery. The initial and target states are both |0>. For circuit
c, p_c(x) is the exact simulated return probability. The optimizer receives

    y(x) = log(max(1e-15, mean_c [1 - p_c(x)])).

Circuit sampling uses SeedSequence([19431, seed]). These are exact
probabilities, so one call costs 40 simulated circuits and zero shots.
Sampling fresh circuits makes the objective stochastic even without shots.

After all recommendations are frozen, an independent 4096-circuit set from
SeedSequence([471931, seed]) measures mean failure probability. The same
holdout circuits evaluate every checkpoint of a run. Standard errors and
all independent circuit counts are recorded separately. The plotted metric
is the natural logarithm of this independent mean failure probability.

## Optimizers and recommendations

Q-STR uses its pulse_calibration preset. The six reference families are
CMA-ES, Nelder--Mead, Powell, differential evolution, Nevergrad OnePlusOne
and simulated annealing. SciPy supplies Powell, Nelder--Mead and DE;
pycma supplies CMA-ES. Simulated annealing uses Gaussian proposals,
Metropolis acceptance and geometric cooling, with the best-so-far return
of Algorithms for Optimization (2019), Algorithm 8.4.

The formal Nelder--Mead comparison initializes a random orthogonal simplex
using seed+173, radius 0.2 and nonadaptive coefficients. Coordinates are
reflected into the physical box. The exact selected parameters of the other
methods are stored in the configuration. DE uses an absolute population
size, rand1bin updates and no polishing. A partial CMA generation consumes
queries but does not replace the native recommendation.

Reference-method checkpoints use strict saved-feedback replay: every replay
query must match the corresponding stored point exactly. No new physical
queries are made. Q-STR checkpoints retain the minimum completed group mean.
An initial point is reported before any native recommendation exists.

## Reproduction

From the bundle directory, with Python 3.10:

~~~sh
python reproduce.py setup --runtimes pulse
python reproduce.py run --problems pulse_calibration --output outputs/pulse
~~~

Use configs/smoke.json for the short integration check. Production validation
uses 4096 circuits; the smoke configuration deliberately reduces this count.
The runner fixes BLAS/OpenMP thread counts before importing NumPy. This is
necessary for pointwise comparison of sensitive stochastic trajectories.

Full-budget validation at seed 7098473 reproduces all 1000 recorded queries,
the native endpoint and independent quality exactly. The final failure
probability is 0.0002317789897920017. See
validation/pulse_ising_full_budget_equivalence.json. The nine-method
integration comparison across pulse and Ising tasks is documented in
validation/pulse_ising_equivalence.json. These checks validate the packaging;
the bundled 30-seed results remain the performance evidence.

## Sources and licenses

The physical model and experiment adapter are implemented in this project.
The reference waveform originates in the archived preparation associated
with [the study's data](https://zenodo.org/records/17352115).
The Clifford decomposition table is adapted from C3 commit
48b7917eb8d84e1471ff9fb911129c354e752dbf. Its Apache-2.0 license and provenance
are included in pulse_support and third_party/pulse.
