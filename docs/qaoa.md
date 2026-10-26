# QAOA MaxCut reproduction

This backend reproduces the five-vertex, depth-five toy experiment in
[A Novel Noise-Aware Classical Optimizer for Variational Quantum Algorithms](https://doi.org/10.1287/ijoc.2024.0578).
The [published software snapshot](https://github.com/INFORMSJoC/2024.0578/tree/42ed201f4f78ed79da5dcf8af1cfc6bd4daf78bd)
supplies the circuit, count statistics, and ANATRA implementation.

## Problem and feedback

The graph is the complete bipartite graph K(3,2), with five vertices and six
edges. Its maximum cut is six. The ten parameters are ordered as

~~~text
x = (beta_1, ..., beta_5, gamma_1, ..., gamma_5).
~~~

Starting from the uniform superposition, each layer applies the ZZ cost
operator with angle gamma, followed by the X mixer with angle beta. The
initial parameter vector is zero. The paid objective is the negative sample
mean of the cut size, evaluated using 1000 fresh measurement counts per query.
ANATRA additionally receives the variance estimate of this sample mean used
by the published implementation. It uses denominator shots squared, rather
than applying an unbiased sample-variance correction.

The circuit is compiled once with transpiler seed 173 and optimization level
zero. A NumPy generator initialized by the experiment seed draws a fresh
simulator seed from [1, 2^31-1) for each query. The reference Sampled.py source
is retained unchanged, while the adapter explicitly supplies the simulator
seed. This preserves the recorded reproducible protocol and avoids the
upstream convenience function's inconsistent seed branch.

The primary saved quality is

~~~text
quality(x) = max(0, 6 - expected_cut(x)).
~~~

Expected cut is calculated by an independent exact statevector implementation
only after the complete paid history and recommendation points have been
written. No true objective, exact derivative or optimum is available to an
optimizer. The displayed negative-cut expectation equals quality minus six.
The initial quality is three. Independent diagnostic points and paid shots
are counted separately.

## Optimizers and settings

| Method | Implementation and retained setting |
|---|---|
| Q-STR | qstr.minimize_with_preset, qaoa_maxcut preset; all paid evaluations include initialization and repetitions |
| ANATRA | Bundled published source; radius 0.1, gradient tolerance 1e-13, identity coordinates, unbounded angles, default model settings |
| Py-BOBYQA | Py-BOBYQA 1.5.0; radius 0.1, bounds [-100,100], objfun_has_noise=True and seek_global_minimum=True, matching scikit-quant 0.8.2's wrapper |
| ImFil | SQImFil 0.3.7; scikit-quant mode, bounds [-100,100], package defaults |
| NOMAD | Official PyNomadBBO 4.4.0 with the recorded mapping of SQNomad 0.2.3 settings; algorithm seed zero |
| SPSA | qiskit-algorithms 0.3.1 SPSA(maxiter=550), package defaults, algorithm seed equal to the experiment seed |

Q-STR and ANATRA are unbounded in the angle coordinates. The artificial
[-100,100] bounds of the three boxed reference methods follow their published
scikit-quant calls. Angles are not wrapped modulo a period during optimization.

The primary comparison caps all methods at 550 paid queries. SPSA's 550
iterations are not 550 function calls: initial calibration and final queries
also consume the cap. The adapter stops before any additional measurement and
retains the last completed iterate. ImFil can similarly request evaluations
beyond its nominal budget; the same external cap is enforced. An interruption
is explicitly recorded and is not represented as a native convergence return.
The reference adapter observes existing optimizer state without extra queries
or changes to the internal update formula.

Q-STR recommends the lowest completed observation-group mean. ANATRA returns
its current-center index. The other algorithms retain their native endpoint
or last completed state. Checkpoints inspect this single optimization run.
NOMAD checkpoint endpoints use exact replay of paid observations, charging zero
additional physical evaluations. A completed endpoint is carried forward when
a solver stops early. The historical minimum of exact objective values is not
used to select a returned point; the publication's retrospective best-value
curves are therefore a different statistic.

## Installation and compatibility

Use the common reproduction setup with CPython 3.10 and requirements/qaoa.txt.
The QAOA requirements lock NumPy 1.26.4, SciPy 1.11.4, COBYQA 1.1.4,
Qiskit 1.2.1, Aer 0.15.1, qiskit-algorithms 0.3.1, NetworkX 3.2.1,
Py-BOBYQA 1.5.0, SQCommon 0.3.2, SQImFil 0.3.7, IBCDFO 1.0.0b1,
ipdb 0.13.13 and PyNomadBBO 4.4.0.

Py-BOBYQA and ImFil are called directly with precisely the wrapper settings
above. This avoids scikit-quant's unconditional SQNomad dependency on Windows.
The recorded NOMAD experiment already used official NOMAD 4.4 instead of
SQNomad's embedded NOMAD 4.0; this package preserves that substitution and
does not claim identical trajectories across those two NOMAD versions.
The packaged SPSA API is the maintained replacement for the removed
qiskit.algorithms API. These are existing reproduction adaptations, not new
changes introduced while packaging.

Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to one before
starting Python, as the supplied launcher does. Multithreaded linear algebra
can change roundoff and subsequently optimizer trajectories. Each NOMAD run
uses a separate process because its C++ state is global.

ANATRA, MINQ, the circuit source, and the compatible fourteen-argument IBCDFO
input check are bundled beneath qaoa_support/vendor. Their upstream licenses
and file hashes are retained in that directory. The input checker is selected
at import time to match the published ANATRA calling convention; its code is
unchanged. Package dependencies are installed normally rather than copied from
an existing environment.

## Verification

The six methods were compared with the frozen research drivers at seed 4073,
80 queries and 1000 shots per query. Under the recorded single-thread setup,
all 480 paired query coordinates, simulator seeds, count histograms and scalar
observations matched exactly. Native endpoint parameters, all four checkpoint
recommendations and their exact physical quality also matched exactly.
The compact record is validation/qaoa_equivalence.json.
This verifies the migration; it is not a new multi-seed performance study.

The full-budget Q-STR run at seed 7098473 was also compared with the archived
550-query experiment. Both stop after 511 paid evaluations. Every coordinate,
count histogram, estimated variance, feedback value, simulator seed and all
22 checkpoint recommendations agree exactly. The final expected-cut gap is
0.12552239374469565 in both runs. See
validation/qaoa_full_budget_equivalence.json for source-record hashes and checks.
