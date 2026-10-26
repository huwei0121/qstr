# Development and reproducibility

## Source layout

Maintain reusable code under src/qstr. Use snake_case for modules and functions,
PascalCase for classes, and UPPER_SNAKE_CASE for constants. Mathematical names such
as x, x0, and n remain appropriate when locally defined. Use descriptive names for
budgets, counters, histories, and coordinate policies.

Name modules after their responsibility. Dates, experiment numbering, comparison
provenance, and candidate labels belong in research records rather than public
identifiers. Update documentation, examples, and affected tests with public API
changes. Use basis factories and objective callables to provide task information.

## Checks

Package tests exercise evaluation accounting, bounds, noisy feedback, recommendation
selection, search coordinates, preset isolation, and native comparison returns.
Distribution checks verify the intended source and documentation inventory.

Use requirements-reproducibility.txt for comparisons with recorded trajectories.
Other supported dependency versions can produce different floating-point results
and optimization trajectories.

## Research assets

The research checkout preserves historical code, scripts, simulation environments,
raw data, and source snapshots. Their original names and hashes are part of the
reproducibility record. They are not imported by qstr or included in its distributions.

Historical imports remain available from the research source tree. New application
code imports qstr. Keep historical result schemas unchanged; convert copies at an
explicit analysis boundary when necessary.

SCIENTIFIC_STATE.md is the checkout's research-status entry. Package documentation
describes software behavior rather than maintaining a second research-status log.

## Distribution

Build with python -m build. Check artifacts with tools/check_distribution.py, then
install the wheel into a fresh environment. Run examples and package tests outside
the research checkout to reveal undeclared local dependencies.

Building a distribution does not upload it or publish a repository.
