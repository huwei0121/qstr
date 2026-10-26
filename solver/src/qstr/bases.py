"""Orthonormal coordinates from pulse grids and circuit layer ordering.

These constructions depend on declared parameter structure and random seeds.
They do not inspect objective values, reference controls, or derivatives.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

BasisFactory = Callable[[int, int], NDArray[np.float64] | None]


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def dct_basis(time_slices: int, modes: int) -> NDArray[np.float64]:
    """Return the first orthonormal DCT-II modes on a uniform time grid.

    Rows correspond to time slices; columns are ordered from the constant
    mode to increasing frequency. ``0 <= modes <= time_slices`` is allowed.
    """
    time_slices = _integer(time_slices, "time_slices")
    modes = _integer(modes, "modes", minimum=0)
    if modes > time_slices:
        raise ValueError("modes must not exceed time_slices")
    times = (np.arange(time_slices) + 0.5)[:, None]
    frequencies = np.arange(modes)[None, :]
    basis = np.sqrt(2.0 / time_slices) * np.cos(np.pi * times * frequencies / time_slices)
    if modes:
        basis[:, 0] = 1.0 / np.sqrt(time_slices)
    return basis


def _orthogonal_completion(
    fixed: NDArray[np.float64], proposal: NDArray[np.float64]
) -> NDArray[np.float64]:
    if proposal.shape[1] == 0:
        return fixed.copy()
    proposal = proposal.copy()
    for _ in range(2):
        proposal -= fixed @ (fixed.T @ proposal)
    singular_values = np.linalg.svd(proposal, compute_uv=False)
    tolerance = np.finfo(float).eps * max(proposal.shape) * singular_values[0]
    if singular_values[-1] <= tolerance:
        raise ValueError("random complement is numerically rank deficient")
    complement, _ = np.linalg.qr(proposal, mode="reduced")
    for _ in range(2):
        complement -= fixed @ (fixed.T @ complement)
    complement, triangular = np.linalg.qr(complement, mode="reduced")
    complement *= np.where(np.diag(triangular) >= 0.0, 1.0, -1.0)
    pivots = np.argmax(np.abs(complement), axis=0)
    complement *= np.where(complement[pivots, np.arange(complement.shape[1])] >= 0.0, 1.0, -1.0)
    return np.column_stack([fixed, complement])


def pulse_basis(
    modes_per_channel: int,
    *,
    time_slices: int = 41,
    core_modes_per_channel: int | None = None,
    seed: int = 0,
    spectral_decay: float = -2.0,
    spectral_scale: float = 8.0,
) -> NDArray[np.float64]:
    """Build coordinates for two quadrature pulse channels.

    Rows are ordered as all in-phase amplitudes followed by all quadrature
    amplitudes. With no ``core_modes_per_channel``, use a block-diagonal DCT
    basis. Otherwise retain that many DCT modes per channel and complete the
    basis with joint Gaussian directions weighted in the DCT domain by
    ``(1 + (frequency / spectral_scale)**2)**(-spectral_decay / 2)``.

    Negative decay emphasizes high-frequency directions. QR preserves their
    span, not their amplitudes. The final basis has ``2*modes_per_channel``
    columns; retained columns precede the random complement.
    """
    time_slices = _integer(time_slices, "time_slices")
    modes_per_channel = _integer(modes_per_channel, "modes_per_channel")
    if modes_per_channel > time_slices:
        raise ValueError("modes_per_channel must not exceed time_slices")
    if core_modes_per_channel is None:
        temporal = dct_basis(time_slices, modes_per_channel)
        basis = np.zeros((2 * time_slices, 2 * modes_per_channel))
        basis[:time_slices, :modes_per_channel] = temporal
        basis[time_slices:, modes_per_channel:] = temporal
        return basis
    core_modes = _integer(core_modes_per_channel, "core_modes_per_channel", minimum=0)
    if core_modes > modes_per_channel:
        raise ValueError("core_modes_per_channel must not exceed modes_per_channel")
    seed = _integer(seed, "seed", minimum=0)
    spectral_decay = float(spectral_decay)
    spectral_scale = float(spectral_scale)
    if not np.isfinite(spectral_decay):
        raise ValueError("spectral_decay must be finite")
    if not np.isfinite(spectral_scale) or spectral_scale <= 0:
        raise ValueError("spectral_scale must be finite and positive")
    temporal = dct_basis(time_slices, core_modes)
    frequencies = np.arange(time_slices, dtype=float)
    if spectral_decay >= 0:
        weights = (1.0 + (frequencies / spectral_scale) ** 2) ** (-0.5 * spectral_decay)
    else:
        log_ratio = np.full(time_slices, -np.inf)
        log_ratio[1:] = np.log(frequencies[1:]) - np.log(spectral_scale)
        log_weights = -0.5 * spectral_decay * np.logaddexp(0.0, 2.0 * log_ratio)
        weights = np.exp(log_weights - np.max(log_weights))
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("spectral weights must remain numerically positive")
    spectral = dct_basis(time_slices, time_slices)
    fixed = np.zeros((2 * time_slices, 2 * core_modes))
    fixed[:time_slices, :core_modes] = temporal
    fixed[time_slices:, core_modes:] = temporal
    extra_modes = 2 * (modes_per_channel - core_modes)
    generator = np.random.default_rng(np.random.SeedSequence([72319, seed]))
    coefficients = generator.standard_normal((2 * time_slices, extra_modes))
    proposal = np.vstack(
        [
            spectral @ (weights[:, None] * coefficients[:time_slices]),
            spectral @ (weights[:, None] * coefficients[time_slices:]),
        ]
    )
    return _orthogonal_completion(fixed, proposal)


def qaoa_layer_basis(layers: int = 5) -> NDArray[np.float64]:
    """Rotate QAOA angles using the singular vectors of layer ordering.

    Input rows are ``[beta_1, ..., beta_p, gamma_1, ..., gamma_p]``. If
    ``L = tril(ones(p,p)) = U Sigma V.T``, return
    ``[[U,U],[-V,V]] / sqrt(2)``. This is a full rotation, not a dimension
    reduction; it exposes coupled angle directions at the zero initial point.
    """
    layers = _integer(layers, "layers")
    left, _, right_transpose = np.linalg.svd(np.tril(np.ones((layers, layers))))
    right = right_transpose.T
    return np.block([[left, left], [-right, right]]) / np.sqrt(2.0)


def state_transfer_basis(
    seed: int,
    *,
    time_slices: int = 100,
    core_modes: int = 8,
    random_modes: int = 4,
) -> NDArray[np.float64]:
    """Combine low-frequency pulse modes and an isotropic random complement."""
    from scipy.fft import dct

    seed = _integer(seed, "seed", minimum=0)
    time_slices = _integer(time_slices, "time_slices")
    core_modes = _integer(core_modes, "core_modes", minimum=0)
    random_modes = _integer(random_modes, "random_modes", minimum=0)
    if not 1 <= core_modes + random_modes <= time_slices:
        raise ValueError("the total number of modes must be in [1, time_slices]")
    fixed = dct(np.eye(time_slices), type=2, norm="ortho", axis=0).T[:, :core_modes]
    proposal = np.random.default_rng(seed).normal(size=(time_slices, random_modes))
    proposal -= fixed @ (fixed.T @ proposal)
    complement = np.linalg.qr(proposal, mode="reduced")[0]
    return np.column_stack((fixed, complement))


def make_basis_factory(preset: str, *, warmup_stages: int = 1) -> BasisFactory | None:
    """Construct the stage-dependent basis associated with a named preset.

    The solver supplies both the stage index and its derived seed. A gate
    calibration preset uses native coordinates and therefore returns ``None``.
    """
    warmup_stages = _integer(warmup_stages, "warmup_stages", minimum=0)
    if preset == "pulse_calibration":

        def pulse_coordinates(stage: int, seed: int) -> NDArray[np.float64]:
            if stage < warmup_stages:
                return pulse_basis(8, seed=seed + 31013)
            return pulse_basis(16, core_modes_per_channel=8, seed=seed + 31013)

        return pulse_coordinates
    if preset == "qaoa_maxcut":
        rotation = qaoa_layer_basis()

        def circuit_coordinates(stage: int, seed: int) -> NDArray[np.float64]:
            return rotation.copy()

        return circuit_coordinates
    if preset == "ising_state_transfer":

        def pulse_coordinates(stage: int, seed: int) -> NDArray[np.float64]:
            return state_transfer_basis(seed)

        return pulse_coordinates
    if preset == "gate_calibration":
        return None
    raise ValueError(f"unknown preset: {preset!r}")
