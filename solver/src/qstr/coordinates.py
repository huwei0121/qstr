"""Smooth local coordinates for finite box constraints."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def anchored_box_map(
    center: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> Callable[[ArrayLike], NDArray[np.float64]]:
    """Construct a smooth box map anchored at an interior point.

    For midpoint ``m``, half-width ``h``, ``a = atanh((center-m)/h)``
    and ``d = h * (1-tanh(a)**2)``, the map is
    ``phi(s) = m + h*tanh(a+s/d)``. In exact arithmetic,
    ``phi(0) = center`` and ``D phi(0) = I``. Consequently,
    ``phi(basis @ z)`` has derivative ``basis`` at the origin.

    The difference form below preserves the anchor exactly in floating-point
    arithmetic. Saturation is limited to the adjacent interior float. Bounds
    and center must be finite vectors with the same shape, and the center
    must be strictly interior.
    """
    center = np.asarray(center, dtype=float).copy()
    lower = np.asarray(lower, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    if center.ndim != 1 or center.size == 0:
        raise ValueError("center must be a nonempty vector")
    if lower.shape != center.shape or upper.shape != center.shape:
        raise ValueError("bounds must have the same shape as center")
    if not all(np.all(np.isfinite(array)) for array in (center, lower, upper)):
        raise ValueError("center and bounds must be finite")
    left, right = center - lower, upper - center
    width = upper - lower
    if np.any(left <= 0) or np.any(right <= 0):
        raise ValueError("the box map requires a strictly interior center")
    if np.any(~np.isfinite(width)):
        raise ValueError("box widths must be representable")
    anchor = 0.5 * (np.log(left) - np.log(right))
    derivative = 2 * (left / width) * right
    if np.any(derivative <= 0):
        raise ValueError("center is too close to a boundary for this map")
    half_width = width / 2
    anchor_tanh = np.tanh(anchor)
    interior_lower = np.nextafter(lower, upper)
    interior_upper = np.nextafter(upper, lower)

    def forward(displacement: ArrayLike) -> NDArray[np.float64]:
        displacement = np.asarray(displacement, dtype=float)
        if displacement.shape != center.shape or not np.all(np.isfinite(displacement)):
            raise ValueError("displacement must be a finite vector matching center")
        with np.errstate(over="ignore"):
            point = center + half_width * (
                np.tanh(anchor + displacement / derivative) - anchor_tanh
            )
        return np.minimum(interior_upper, np.maximum(interior_lower, point))

    return forward
