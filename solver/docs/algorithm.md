# Algorithm

Q-STR minimizes an expectation-valued scalar objective under an evaluation budget:

    minimize f(x) = E[feedback(x, random_state)],  x in X.

The feasible set X is either unconstrained or a finite box. The solver observes
scalar feedback and uses no reference optimum or exact derivative.

## Search coordinates

At a stage center x_c, a basis B with B.T @ B = I defines x = x_c + B @ z.
For bounded structured problems, an anchored hyperbolic-tangent map satisfies
phi(0) = x_c and Dphi(0) = B in the absence of floating-point clipping.
Native coordinates pass bounds directly to COBYQA. A new basis can introduce
directions outside the previous stage's search space.

## Quadratic models

COBYQA maintains an interpolation set and a quadratic model:

    m(s) = c + g.T @ s + 0.5 * s.T @ H @ s.

With fewer observations than quadratic coefficients, the minimum Hessian-change
criterion retains previously inferred curvature subject to updated interpolation
conditions. The numerical engine handles trust-region steps, geometry improvement,
radius and resolution changes, and model resets using the same interpolation set.

The reduction ratio guides radius updates. A trial observation can enter the model
even when its value does not improve. Point selection and radius adjustment are
distinct operations.

## Repetitions and budget

An observation group averages a fixed number of repeated calls at one point.
Every call consumes one unit of max_evaluations. The final group can contain fewer
calls when the budget ends.

Each stage starts from the smallest recorded group mean. A stage can stop at its
evaluation cap, a resolution stop, or a configured stagnation criterion. A new
stage initializes a new model. Historical observations remain available for
recommendation, but are not automatically inserted into a new interpolation model.

If a planned stage would leave too little budget to initialize another model,
it receives the remaining budget before COBYQA starts. The active model is then
preserved throughout that call.

## Randomness and noisy observations

The solver seed controls search directions. The objective owns its measurement
random stream; reproducible simulation should seed these streams separately.

Repetitions are fixed by configuration. The current implementation does not infer
a noise level or implement confidence-based acceptance. Small construction scales
can amplify observation errors, especially in curvature. Independent validation
of a recommended point should be accounted for separately.

## Implementation

Coordinate arithmetic, random draws, stage scheduling, and evaluation order were
checked against the frozen research implementation. Renamed options express the
same quantities. The numerical engine remains COBYQA 1.1.4.
