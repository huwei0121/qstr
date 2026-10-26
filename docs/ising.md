# Ising state transfer with coupling noise

This problem follows Section 3.2 of M. Rossignolo et al.,
[QuOCS: The Quantum Optimal Control Suite](https://arxiv.org/abs/2212.11144).
The bundle includes the full QuOCS implementation at commit
3d67403cad44ea03450132fec25afa9114b6b2f1, its installation metadata and its
Apache-2.0 license. The default comparison uses full dCRAB and Q-STR.

## Model and optimization variables

For five spins with periodic indices, the Hamiltonian is

    H(t; g) = -sum_j Z_j Z_(j+1) - g sum_j Z_j Z_(j+2) + u(t) sum_j X_j.

The initial state is |00000>, the target is |11111>, and the total evolution
time is one. The pulse u has 100 piecewise-constant slices. Normalized
variables satisfy x in [-1,1]^100 and u_j = 1000*x_j. Both methods start from
the zero pulse. Each paid function evaluation independently draws

    g = 2 + sqrt(0.1)*Normal(0,1)

and returns the state-transfer infidelity 1 - |<11111|U(x;g)|00000>|^2.
The Gaussian variance is 0.1. The objective RNG uses
SeedSequence([seed, 71855]). One call costs one physical propagation;
there is no finite-shot noise in this benchmark.

Upstream QuOCS constructs the Hamiltonians. The adapter projects them onto
the exact cyclic-invariant subspace, which contains both initial and target
states and is preserved by every Hamiltonian term. For five spins its
dimension is eight, compared with 32 in the full Hilbert space. Batched
Hermitian eigendecompositions then form the exact propagator for each slice.
This changes computational cost, not the modeled state-transfer problem.
The optimizer receives only scalar sampled infidelity.

## Complete dCRAB implementation

The baseline invokes QuOCS Optimizer and its complete dCRAB implementation.
It uses three superiterations, five random Fourier frequencies (ten sine/
cosine coefficients), frequency range [0.01,5], and amplitude variation 10.
The inner optimizer is adaptive Nelder--Mead. Basis changes and the native
re-evaluation mechanism run inside QuOCS. This is not a single fixed-basis
Nelder--Mead solve.

The exact starting configuration is shipped as ising_support/dcrab.json;
each run saves its effective configuration. The common cap is 2000 paid
propagations and includes repeated measurements. QuOCS receives the fixed
reported uncertainty 0.1; it does not receive the true variance of the
simulated infidelity. The configured frtol=0.001 is not consumed by this
upstream revision. Its effective fatol is 1e-100 and xatol is 1e-14; the
change-based stopping settings remain those in the shipped configuration.

Q-STR uses its ising_state_transfer preset, including three repeated
observations per group, smooth/random 12-dimensional search spaces and a
500-evaluation nominal stage budget. Both methods keep their own native
recommendation rules. dCRAB checkpoints record best controls from completed
inner-routine calls; Q-STR records improvements in completed group means.
No independent metric selects any recommendation.

## Independent validation and plotted metric

The reported quality is expected infidelity over the Gaussian coupling.
After point selection is finished, 128-node Gauss--Hermite quadrature
approximates this expectation:

    E[f(x;g)] = sum_i w_i f(x; 2+sqrt(0.2)*z_i) / sqrt(pi).

A separate 64-node endpoint calculation measures quadrature convergence.
These validation propagations are recorded separately from paid search
queries. The plotted fidelity percentage is 100*(1 - expected_infidelity).

## Reproduction and validation

From the bundle directory, with Python 3.10:

~~~sh
python reproduce.py setup --runtimes pulse
python reproduce.py run --problems ising_state_transfer --output outputs/ising
~~~

Both pulse and Ising tasks use the pulse environment. Its requirements
install the bundled QuOCS source locally. The runner sets numerical thread
counts to one before importing numerical libraries. The reduced quadrature
in configs/smoke.json is for software integration checks only.

At seed 7098473, all 2000 Q-STR query points, feedback values, the native
return and independent quality match the recorded comparison exactly.
Final expected infidelity is 0.06966379026215577. Full-budget evidence is
in validation/pulse_ising_full_budget_equivalence.json; nine short method comparisons
are recorded in validation/pulse_ising_equivalence.json.
