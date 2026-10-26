# Verification of the reproduction package

The benchmark implementation calls the packaged qstr solver and installed
reference methods directly. All four backends were compared with the retained
research execution, using its physical feedback and recommendation conventions.

An independent directory outside the research checkout was used to create all
four Python 3.10 environments on Windows x64. Installation, dependency checks,
all 17 reduced-budget problem/method combinations, and four full-budget Q-STR
runs completed successfully. The Windows dependency constraints were reapplied
and checked in these environments. Fourteen focused tests cover analysis,
launching, comparison, QAOA sampling, and independent physical computations.
An interrupted-attempt check preserved old files, reran the case, and skipped
the completed case on the next resume without changing its record.

The full-budget Q-STR check uses seed 7098473 for every problem:

| Problem | Query cap | Actual paid queries | Final independent error |
| --- | ---: | ---: | ---: |
| Pulse calibration | 1000 | 1000 | 0.0002317789897920017 |
| QAOA MaxCut | 550 | 511 | 0.12552239374469565 |
| Gate calibration | 100 | 98 | 0.2254066546256517 |
| Ising state transfer | 2000 | 2000 | 0.06966379026215577 |

Under the recorded dependency versions and single-thread setup, all 3609 paid
parameter vectors, feedback values and final recommendations match the original
stored runs exactly. Saved checkpoint physical quality also agrees exactly.
QAOA's 511 and gate calibration's 98 calls are the same early stops as before.
These are migration checks, not a new optimization improvement.

Additional small-budget comparisons cover every reference method. QAOA's six
methods account for 480 paired queries; pulse/Ising's nine combinations account
for 360. Gate calibration checks physical probabilities, shot counts and CMA
returns at complete and partial population budgets. Compact evidence is saved
under validation, including source hashes.

Independent physical checks compare the pulse propagator with adaptive DOP853,
Ising's invariant subspace with full 32-dimensional propagation, and QAOA with an
independent statevector computation. Statistical checks reaggregate all 510 seed
records and compare every previously plotted coordinate, mean and quartile.

## Floating-point execution

reproduce.py sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
TF_NUM_INTRAOP_THREADS and TF_NUM_INTEROP_THREADS to one before the worker imports
numerical libraries. The solver and measurement streams are seeded separately.
Changing the BLAS reduction order can introduce roundoff which subsequently
changes optimizer decisions; a direct multithreaded call need not replay the
same trajectory. Tests of exact replay refer to the specified environment and
platform. Other platforms can verify model tolerances and statistical behavior
without assuming bitwise equality across different numerical libraries.

The four-problem figure uses the same array reduction order as the original
analysis. The initial-to-first-checkpoint segment is a dashed visual guide.
Software checks and the reduced-budget smoke configuration do not replace the
full 510-run configuration when estimating performance.
