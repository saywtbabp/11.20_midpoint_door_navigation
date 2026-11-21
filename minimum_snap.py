"""Piecewise minimum-snap trajectories inside convex planar corridors."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np


_DEGREE = 7
_POSITION_TOL = 1e-8
_BINOMIAL = tuple(np.array([comb(n, i) for i in range(n+1)])
                  for n in range(_DEGREE+1))
_POWERS = tuple(np.arange(n+1) for n in range(_DEGREE+1))
# Integral of B_i^3(s) B_j^3(s) over [0, 1].
_SNAP_GRAM = np.array([[comb(3, i)*comb(3, j)/(7*comb(6, i+j))
                        for i in range(4)] for j in range(4)])
_SNAP_FACTOR = np.linalg.cholesky(_SNAP_GRAM).T


def _split_half(polygon):
    left, right = [polygon[0]], [polygon[-1]]
    for _ in range(len(polygon)-1):
        polygon = (polygon[1:]+polygon[:-1])*.5
        left.append(polygon[0])
        right.append(polygon[-1])
    return np.array(left), np.array(right[::-1])


def _subdivision_matrix(degree):
    polygons = [np.eye(degree+1)]
    for _ in range(3):
        polygons = [half for polygon in polygons for half in _split_half(polygon)]
    return np.concatenate(polygons)


_SUBDIVISION = {degree: _subdivision_matrix(degree) for degree in (4, 5, 6)}


@dataclass(frozen=True)
class MinimumSnapTrajectory:
    """A time-indexed sequence of seventh-degree Bernstein control polygons."""

    control_points: np.ndarray
    durations: np.ndarray
    _ends: np.ndarray = field(init=False, repr=False)
    _derivatives: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_ends", np.cumsum(self.durations))
        derivative = self.control_points
        derivatives = [derivative]
        for degree in (7, 6, 5):
            derivative = degree*np.diff(derivative, axis=1)/self.durations[:, None, None]
            derivatives.append(derivative)
        object.__setattr__(self, "_derivatives", tuple(derivatives))

    @property
    def duration(self) -> float:
        return float(self._ends[-1])

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return position, velocity, acceleration and jerk at elapsed time t."""
        if t <= 0.:
            return tuple(value[0, 0].copy() for value in self._derivatives)
        if t >= self.duration:
            return tuple(value[-1, -1].copy() for value in self._derivatives)
        segment = int(np.searchsorted(self._ends, t, side="right"))
        offset = self._ends[segment-1] if segment else 0.
        s = (t-offset)/self.durations[segment]
        return tuple((_BINOMIAL[degree]*s**_POWERS[degree] *
                      (1.-s)**_POWERS[degree][::-1]) @ value[segment]
                     for degree, value in zip((7, 6, 5, 4), self._derivatives))

    def derivative_bounds(self) -> tuple[float, float, float]:
        """Conservative speed, acceleration and jerk norms over the trajectory.

        Subdivision tightens the convex-hull bound without sampling the curve.
        """
        return tuple(float(np.linalg.norm(_SUBDIVISION[degree] @ value, axis=-1).max())
                     for degree, value in zip((6, 5, 4), self._derivatives[1:]))


def _control_map(points, durations, velocity, acceleration, corridors):
    """Split the control polygons into known and free endpoint derivatives."""
    count = len(durations)
    scale = float(np.median(durations))
    # Scale free derivatives in units of scale^k to balance orders numerically.
    controls = np.repeat(points[:-1, None, :], 8, axis=1)
    controls[:, 4:, :] = points[1:, None, :]
    mapping = np.zeros((count, 8, 1+3*(count-1)))
    first = durations[0]*velocity/7.
    second = durations[0]**2*acceleration/42.
    controls[0, 1:4] += (first, 2.*first+second, 3.*first+3.*second)

    # Jerk at the start is free. If the nominal zero-jerk start lies outside,
    # choose its preceding control point as a feasible baseline for this point.
    normal, offset = corridors[0]
    if np.any(normal @ controls[0, 3] > offset+_POSITION_TOL):
        controls[0, 3] = controls[0, 2]
    mapping[0, 3, 0] = (durations[0]/scale)**3/210.
    for knot in range(1, count):
        col = 1+3*(knot-1)
        after, before = durations[knot]/scale, durations[knot-1]/scale
        for segment, index, ratio, sign in ((knot, (1, 2, 3), after, 1.),
                                             (knot-1, (6, 5, 4), before, -1.)):
            v, a, j = sign*ratio/7., ratio**2/42., sign*ratio**3/210.
            mapping[segment, index, col:col+3] = ((v, 0., 0.), (2.*v, a, 0.), (3.*v, 3.*a, j))
    return controls, mapping, scale


def _bounded_least_squares(target, normals, offsets):
    """Project the unconstrained optimum onto half-spaces from feasible zero."""
    if not len(normals) or np.all(normals @ target <= offsets+1e-10):
        return target
    point = np.zeros_like(target)
    active = []
    for _ in range(16*(len(target)+1)):
        residual = target-point
        if active:
            active_normals = normals[active]
            basis = np.linalg.qr(active_normals.T, mode="reduced")[0]
            direction = residual-basis @ (basis.T @ residual)
        else:
            direction = residual
        length = np.linalg.norm(direction)
        if length <= 1e-10*max(1., np.linalg.norm(residual)):
            if not active:
                break
            multipliers = np.linalg.lstsq(active_normals.T, residual, rcond=None)[0]
            worst = int(np.argmin(multipliers))
            if multipliers[worst] >= -1e-10:
                break
            active.pop(worst)
            continue
        slope = normals @ direction
        can_block = slope > 1e-10*length
        can_block[active] = False
        blockers = np.flatnonzero(can_block)
        slack = offsets[blockers]-normals[blockers] @ point
        ratios = slack/slope[blockers]
        closest = int(np.argmin(ratios)) if len(ratios) else None
        if closest is None or ratios[closest] >= 1.:
            point = point+direction
            continue
        point = point+max(0., ratios[closest])*direction
        active.append(int(blockers[closest]))
    return point


def solve_minimum_snap(points, durations, start_velocity, start_acceleration,
                       corridors) -> MinimumSnapTrajectory | None:
    """Fit all waypoints with shared physical derivatives up to jerk.

    Segment i must stay in its convex corridor A_i @ p <= b_i. The Bernstein
    convex-hull constraints apply to its entire curve. The terminal state has
    zero velocity, acceleration and jerk. A failed feasibility check returns
    None so the caller can try other timings or a different geometric route.
    """
    waypoints = np.asarray(points, dtype=float)
    times = np.asarray(durations, dtype=float)
    v0, a0 = np.asarray(start_velocity, dtype=float), np.asarray(start_acceleration, dtype=float)
    if (waypoints.ndim != 2 or waypoints.shape[1] != 2 or len(waypoints) < 2
            or times.shape != (len(waypoints)-1,) or v0.shape != (2,) or a0.shape != (2,)
            or len(corridors) != len(times)):
        raise ValueError("expected planar waypoints and one duration and corridor per segment")
    if any(not np.isfinite(value).all() for value in (waypoints, times, v0, a0)) or np.any(times <= 0.):
        return None

    normalized = []
    for a, b in corridors:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or b.shape != (len(a),):
            raise ValueError("each corridor must be represented as A[M, 2], b[M]")
        sizes = np.linalg.norm(a, axis=1)
        if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(sizes <= 0.):
            return None
        normalized.append((a/sizes[:, None], b/sizes))

    controls, mapping, scale = _control_map(waypoints, times, v0, a0, normalized)
    if any(np.any(control @ a.T > b+_POSITION_TOL)
           for control, (a, b) in zip(controls, normalized)):
        return None

    # The integral of squared snap scales as T^-7 in normalized time.
    # Each block is a weighted, fourth Bernstein control-point difference.
    weight = (scale/times)**3.5
    system = (weight[:, None, None]*(_SNAP_FACTOR @ np.diff(mapping, n=4, axis=1))).reshape(
        (-1, mapping.shape[-1]))
    known = (weight[:, None, None]*(_SNAP_FACTOR @ np.diff(controls, n=4, axis=1))).reshape((-1, 2))
    col_scales = np.linalg.norm(system, axis=0)
    if np.any(col_scales == 0.) or not np.isfinite(system).all() or not np.isfinite(known).all():
        return None
    try:
        q, r = np.linalg.qr(system/col_scales)
        lift = np.linalg.solve(r, np.eye(len(r)))/col_scales[:, None]
    except np.linalg.LinAlgError:
        return None

    # In these coordinates the objective is squared distance to target.
    target = (-q.T @ known).reshape(-1)
    shapes = mapping @ lift
    normals, offsets = [], []
    for a_b, baseline, shape in zip(normalized, controls, shapes):
        a, b = a_b
        normals.append(np.einsum("ki,md->kmid", shape, a).reshape((8*len(a), target.size)))
        offsets.append((b-baseline @ a.T).reshape(-1))
    linear, bound = np.concatenate(normals), np.concatenate(offsets)
    lengths = np.linalg.norm(linear, axis=1)
    variable = lengths > 1e-14
    # The baseline is feasible to numerical tolerance, so tiny negative slacks
    # at its boundary are treated as zero by the active-set iteration.
    updated = _bounded_least_squares(target, linear[variable]/lengths[variable, None],
                                     np.maximum(0., bound[variable])/lengths[variable])
    candidate = controls+shapes @ updated.reshape((-1, 2))
    if np.isfinite(candidate).all() and all(np.all(control @ a.T <= b+_POSITION_TOL)
                                         for control, (a, b) in zip(candidate, normalized)):
        controls = candidate
    return MinimumSnapTrajectory(controls, times.copy())
