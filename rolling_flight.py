"""Rolling, corridor-certified references for the free-flight actions."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from flight_corridors import FLIGHT_MARGIN, build_corridors, certify_controls
from minimum_snap import solve_minimum_snap
from simulator import MIXER

FLIGHT_PHASES = frozenset(("fly", "approach", "bypass", "cross", "release", "retreat"))
ROUTE_LATERAL_M = .35
HANDOFF_LATERAL_M = .55
ROUTE_END_MARGIN_M = .40
WINDOW_M = 10.
REFRESH_S = .85
RETRY_S = .25
TRACK_ERROR_M = .17
MAX_ACCEL = 1.85
MAX_JERK = 6.
SPEEDS = {"fly": 1.50, "approach": .78, "bypass": 1.30, "cross": 1.20,
          "release": 1.02, "retreat": .68}


def _right_controls(control, u):
    """Control polygon for the suffix of a Bézier segment after parameter u."""
    level = np.asarray(control)
    right = [level[-1]]
    while len(level) > 1:
        level = (1.-u)*level[:-1] + u*level[1:]
        right.append(level[-1])
    return np.asarray(right[::-1])


@dataclass
class ScheduledFlight:
    curve: object
    start: float
    revision: int
    seamless: tuple
    extend: bool
    braking: bool = False

    def reference(self, t):
        return self.curve.sample(t-self.start)

    def remaining(self, t):
        elapsed = max(0., t-self.start)
        controls = self.curve.control_points
        for i, duration in enumerate(self.curve.durations):
            if elapsed < duration:
                return (_right_controls(controls[i], elapsed/duration), *controls[i+1:])
            elapsed -= duration
        return (controls[-1][-1:],)


def _window(position, waypoints):
    points = [np.asarray(position)]
    remaining = WINDOW_M
    extend = False
    for waypoint in waypoints:
        length = float(np.linalg.norm(waypoint-points[-1]))
        if length < .025 and len(points) > 1:
            continue
        if length < 1e-6:
            continue
        if length > remaining:
            points.append(points[-1] + (waypoint-points[-1])*remaining/length)
            extend = True
            break
        points.append(waypoint)
        remaining -= length
        if remaining < .10 or len(points) == 4:
            extend = float(np.linalg.norm(points[-1]-waypoints[-1])) > .025
            break
    return np.asarray(points), extend


class RollingFlight:
    def __init__(self, arena, mass):
        self.arena = arena
        self.mass = mass
        self.active = None
        self.leaves_key = None
        self.next_attempt = 0.
        self.mode = "inactive"
        self.failure_cause = ""
        self.hold = None
        self.plans = 0
        self.extensions = 0
        self.rechecks = 0
        self.invalidations = 0
        self.failures = 0
        self.brakes = 0
        self.solve_ms = []
        self.preview = ()

    def _nominal_motor_peak(self, acceleration, jerk):
        # Fixed-height nominal collective and a reserve for the attitude-rate
        # feedback. Motor commands in the closed loop are audited separately.
        collective = self.mass*np.hypot(9.81, acceleration)
        rate_reserve = np.abs(MIXER[:, 1:]) @ (np.array([.16, .16, .09])*jerk/9.81)
        return float(np.max(MIXER[:, 0]*collective + rate_reserve))

    def _solve(self, points, velocity, acceleration, leaves, speed, *, lateral_margin=ROUTE_LATERAL_M):
        corridors = build_corridors(points, self.arena, leaves, margin=FLIGHT_MARGIN)
        if corridors is None:
            return None, "corridor"
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        for index, (a, b) in enumerate(corridors):
            # Snap alone can send a curve far across a wide room while
            # joining two short gate legs. Bound its deviation from the route.
            along = segments[index]/lengths[index]
            across = np.array([-along[1], along[0]])
            directions = np.vstack((across, -across, -along, along))
            bounds = directions @ points[index]
            bounds += (lateral_margin, lateral_margin, ROUTE_END_MARGIN_M,
                       lengths[index]+ROUTE_END_MARGIN_M)
            corridors[index] = np.vstack((a, directions)), np.r_[b, bounds]
        speed = max(speed, float(np.linalg.norm(velocity))*1.05)
        acceleration_limit = max(MAX_ACCEL, float(np.linalg.norm(acceleration))*1.05)
        durations = np.maximum(.55, 1.5*lengths/speed)
        for _ in range(6):
            curve = solve_minimum_snap(points, durations, velocity, acceleration, corridors)
            if curve is None:
                return None, "boundary"
            vmax, amax, jmax = curve.derivative_bounds()
            scale = max(vmax/speed, np.sqrt(amax/acceleration_limit), np.cbrt(jmax/MAX_JERK))
            if self._nominal_motor_peak(amax, jmax) > 9.0:
                scale = max(scale, 1.16)
            if scale <= 1.001:
                return curve, ""
            durations *= max(1.06, 1.035*scale)
        return None, "dynamic"

    def _brake(self, t, position, velocity, acceleration, revision, leaves):
        speed = float(np.linalg.norm(velocity))
        if speed < .07:
            return None
        direction = velocity/speed
        distance = max(.08, .55*speed + speed*speed/(2*MAX_ACCEL))
        finish = position + distance*direction
        curve, _ = self._solve(np.vstack((position, finish)), velocity, acceleration, leaves,
                               max(.5, speed*1.08))
        if curve is None:
            return None
        self.brakes += 1
        return ScheduledFlight(curve, t, revision, (), False, braking=True)

    def _set(self, scheduled, mode):
        self.active = scheduled
        self.mode = mode
        self.failure_cause = ""
        self.hold = None
        self.preview = (tuple(tuple(scheduled.curve.sample(t)[0])
                              for t in np.linspace(0., scheduled.curve.duration, 35))
                        if scheduled else ())

    def can_pass(self, t, waypoint):
        if self.active is None or waypoint is None:
            return False
        elapsed = t-self.active.start
        return any(abs(elapsed-stamp) < .55 and np.linalg.norm(waypoint-point) < .015
                   for point, stamp in self.active.seamless)

    def reference(self, t):
        if self.active is not None:
            return self.active.reference(t)
        if self.hold is not None:
            return self.hold, np.zeros(2), np.zeros(2), np.zeros(2)
        return None

    def update(self, t, position, velocity, acceleration, revision, waypoints, phase, leaves, *,
               separating=False):
        if phase not in FLIGHT_PHASES or not waypoints:
            self._set(None, "inactive")
            return
        leaves = tuple(leaves)
        key = tuple((leaf.gate.name, leaf.side, round(leaf.angle, 6)) for leaf in leaves)
        old = self.active
        if old is not None and key != self.leaves_key:
            self.rechecks += 1
            if not all(certify_controls(piece, self.arena, leaves, margin=FLIGHT_MARGIN)
                       for piece in old.remaining(t)):
                self.invalidations += 1
                self._set(None, "invalidated")
                self.next_attempt = t
                old = None
        self.leaves_key = key
        same_path = old is not None and old.revision == revision
        previous = old.reference(t) if old is not None else None
        error = float(np.linalg.norm(position-previous[0])) if previous is not None else 0.
        time_left = old.start + old.curve.duration - t if old is not None else 0.
        extension = (same_path and old.extend
                     and time_left < max(REFRESH_S, .45*old.curve.duration))
        retry_after_brake = old is not None and old.braking and time_left <= .01
        if same_path and not extension and not retry_after_brake and error < TRACK_ERROR_M:
            return
        if t < self.next_attempt and (old is None or same_path):
            return
        recovering = self.mode == "holding" and self.hold is not None
        if recovering and (np.linalg.norm(velocity) > .11 or np.linalg.norm(position-self.hold) > .10):
            return

        # The current contact-to-free handoff first follows the verified separating
        # direction. A normal collision-free corridor starts outside the leaf.
        if (old is None and separating
                and not certify_controls(position.reshape(1, 2), self.arena, leaves, margin=FLIGHT_MARGIN)):
            self._set(None, "separating")
            return

        started = perf_counter()
        use_previous = previous is not None and error < .09 and not old.braking
        initial_acceleration = np.clip(acceleration, -.45, .45) if recovering else acceleration
        state = previous[:3] if use_previous else (position, velocity, initial_acceleration)
        points, extend = _window(state[0], waypoints)
        # Let a moving approach turn gradually from the incoming tangent while
        # keeping the wall and leaf half-spaces in the corridor.
        lateral_margin = (HANDOFF_LATERAL_M if phase == "approach" and use_previous
                          and not same_path and len(points) == 2 else ROUTE_LATERAL_M)
        curve, why = (self._solve(points, state[1], state[2], leaves, SPEEDS[phase],
                                  lateral_margin=lateral_margin)
                      if len(points) > 1 else (None, "at_target"))
        if curve is None and use_previous and why != "at_target":
            points, extend = _window(position, waypoints)
            if len(points) > 1:
                curve, why = self._solve(points, velocity, acceleration, leaves, SPEEDS[phase],
                                         lateral_margin=lateral_margin)
        if curve is None and why == "boundary" and old is None and phase == "release":
            # The residual motion after contact can need extra room to turn.
            # Retry the route-band constraint before resorting to a brake.
            curve, why = self._solve(points, velocity, initial_acceleration, leaves, SPEEDS[phase],
                                     lateral_margin=HANDOFF_LATERAL_M)
        self.solve_ms.append((perf_counter()-started)*1000.)
        if curve is not None:
            knots = np.cumsum(curve.durations)
            seamless = tuple(zip(points[1:-1], knots[:-1]))
            self._set(ScheduledFlight(curve, t, revision, seamless, extend), "tracking")
            self.next_attempt = t + .10
            self.plans += 1
            self.extensions += int(extension)
            return

        self.next_attempt = t + RETRY_S
        if why != "at_target":
            self.failures += 1
        if same_path and error < TRACK_ERROR_M and time_left > .10 and not old.braking:
            self.mode = "validated_prefix"
            self.failure_cause = why
            return
        brake = self._brake(t, position, velocity, acceleration, revision, leaves)
        previous_hold = self.hold if recovering else None
        self._set(brake, "braking" if brake else ("arrived" if why == "at_target" else "holding"))
        self.failure_cause = why
        if brake is None:
            self.hold = position.copy() if previous_hold is None else previous_hold

    def diagnostics(self):
        return dict(plans=self.plans, horizon_extensions=self.extensions, geometry_rechecks=self.rechecks,
                    invalidations=self.invalidations, solve_failures=self.failures, brakes=self.brakes,
                    median_solve_ms=round(float(np.median(self.solve_ms)), 3) if self.solve_ms else None,
                    max_solve_ms=round(max(self.solve_ms), 3) if self.solve_ms else None)
