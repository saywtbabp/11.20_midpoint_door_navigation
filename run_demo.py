"""Open this file in PyCharm and click Run with the Anaconda mujoco environment."""
from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from time import perf_counter, sleep

# PyCharm can launch a Conda interpreter without activating its environment.
# On Windows, NumPy's BLAS loads native libraries through PATH.
if sys.platform == "win32":
    conda_bin = Path(sys.prefix) / "Library" / "bin"
    if conda_bin.is_dir():
        inherited_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join((str(conda_bin), inherited_path))

import mujoco.viewer
import numpy as np

from door_geometry import segment_clear
from door_resistance import ThrustResidual
from localization import Localization
from planner import Navigator, PUSH_ACCEL_LIMIT_MPS2
from rolling_flight import RollingFlight
from scene import CAGE_RADIUS, make_scene
from simulator import ALTITUDE, DT, RAYS, Simulator, rotor_control
from viewer_pause import ViewerPause

SHOW_VIEWER = True
SHOW_DASHBOARD = True
USE_MINIMUM_SNAP = True
REALTIME_FACTOR = 1.0
EFFORT_WEIGHT_M_PER_J = 1/.030
PUSH_FORCE_RAMP_N_PER_S = 1.20
SEED = 7
RANGE_NOISE_M = .003
SCAN_STEPS = 25
OUTPUT_DIR = Path(__file__).with_name("runs")
MODE = "auto"
UNLOAD_BLEND_S = .16


def _measured_clearance(sim, obstacle_ids):
    """Offline cage-envelope clearance using only the physical scene."""
    data = sim.data
    dx = sim.imu.xpos[:2] - data.geom_xpos[obstacle_ids, :2]
    xy_axes = data.geom_xmat[obstacle_ids].reshape(-1, 3, 3)[:, :2, :2]
    local = np.einsum("ni,nij->nj", dx, xy_axes)
    outside = np.maximum(abs(local) - sim.model.geom_size[obstacle_ids, :2], 0.)
    return float(np.min(np.linalg.norm(outside, axis=1))) - CAGE_RADIUS


def run(*, view=False, dashboard=False, check=False, seed=SEED,
        noise=RANGE_NOISE_M, mode=MODE, trace=None, summary_file=None,
        force_ramp=PUSH_FORCE_RAMP_N_PER_S, optimize_trajectory=USE_MINIMUM_SNAP):
    arena, fixtures, names = make_scene()
    sim = Simulator(arena, fixtures, seed, noise)
    pose = Localization(arena, ALTITUDE)
    nav = Navigator(arena, effort_weight=EFFORT_WEIGHT_M_PER_J, check=check, mode=mode,
                    show_routes=view and dashboard, force_ramp=force_ramp,
                    smooth_paths=optimize_trajectory)
    flight = RollingFlight(arena, sim.mass) if optimize_trajectory else None
    trajectory_errors = []
    free_clearance = float("inf")
    motor_peak = 0.
    motor_saturation_steps = 0
    unloading_started = -1.
    contact_command = np.zeros(2)
    load = ThrustResidual(sim.mass)
    panel_ids = {name: sim.model.geom(name+"_panel").id for name in names}
    panel_names = {index: name for name, index in panel_ids.items()}
    wall_ids = {sim.model.geom(f"wall_{index}").id for index in range(len(arena.walls))}
    watch_ids = set(panel_ids.values()) | wall_ids
    obstacle_ids = np.asarray(sorted(watch_ids))
    touches = Counter()
    passage_contact = Counter()
    passage_contact_last = {}
    passage_started = 0.
    angles = {}
    target = arena.start
    thrust = np.zeros(4)
    motion = np.zeros(3)
    limited_steps = 0
    distance = 0.
    previous = arena.start
    poor_scans = 0
    reason = "time_limit"
    pause = ViewerPause() if view else None
    if view and dashboard:
        from live_dashboard import Dashboard
        board = Dashboard(arena, names)
    else:
        board = None
    context = (mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=pause.on_key)
               if view else nullcontext())
    with context as viewer:
        if viewer:
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -68
            viewer.cam.distance = 14
        previous_event = 0
        frame_started = perf_counter()
        for step in range(round(240 / DT)):
            t = step*DT
            if viewer and not pause.ready(viewer, board.pump if board else None):
                reason = "viewer_closed"
                break
            acc, gyro = sim.inertial()
            if step:
                near_hover = (np.linalg.norm(motion[:2]) < .15 and abs(motion[2]) < .35
                              and nav.phase not in ("push", "boost", "release", "retreat"))
                pose.predict(acc, gyro, DT, near_hover=near_hover)
            load.observe(pose.rotation, acc, thrust)
            if step % SCAN_STEPS == 0:
                reaction = load.take()
                points = pose.correct(RAYS, sim.scan(), DT*SCAN_STEPS)
                p = pose.position[:2]
                distance += np.linalg.norm(p-previous)
                previous = p.copy()
                if np.linalg.norm(p-arena.goal) < .15 and np.linalg.norm(pose.velocity[:2]) < .24:
                    reason = "goal"
                    break
                poor_scans = poor_scans + 1 if pose.hits < 10 else 0
                if not .5 < pose.position[2] < 2.2 or poor_scans > 16:
                    reason = "localization_or_flight_envelope"
                    break
                preceding_phase = nav.phase
                fly_through = (flight.can_pass(t, nav.path[0]) if flight is not None and nav.path else False)
                target = nav.decide(p, pose.velocity[:2], points, reaction, t,
                                    planar_command=motion[:2],
                                    output_limited=limited_steps >= SCAN_STEPS//2, fly_through=fly_through)
                limited_steps = 0
                if (flight is not None and preceding_phase in ("push", "boost")
                        and nav.phase in ("release", "retreat")):
                    unloading_started, contact_command = t, motion[:2].copy()
                clearance = None
                if nav.phase in ("fly", "approach", "cross", "bypass"):
                    clearance = _measured_clearance(sim, obstacle_ids)
                    free_clearance = min(free_clearance, clearance)
                if flight is not None:
                    leaves = dict(nav.panels)
                    if nav.attempt is not None:
                        leaves[nav.active] = (nav.attempt.leaf,)
                    measured_acceleration = (pose.rotation @ acc)[:2]
                    separating = (nav.attempt is not None and nav.phase in ("release", "retreat")
                                  and nav.path and (
                                      segment_clear(p, nav.path[0], nav.attempt.leaf, separating=True)
                                      or segment_clear(p, nav.path[0], nav.attempt.leaf)))
                    flight.update(t, p, pose.velocity[:2], measured_acceleration,
                                  nav.path_revision, nav.path, nav.phase,
                                  (leaf for name in sorted(leaves) for leaf in leaves[name]),
                                  separating=separating)
                    if flight.failure_cause == "corridor" and flight.mode in ("holding", "braking"):
                        target = nav.repair_door_path(p)
                    reference = flight.reference(t)
                    if reference is not None and flight.active is not None:
                        trajectory_errors.append(float(np.linalg.norm(p-reference[0])))
                if nav.phase in ("release", "cross") and nav.phase != preceding_phase:
                    passage_started = t
                attempt = nav.attempt
                if attempt and nav.phase in ("push", "boost"):
                    truth = np.rad2deg(sim.data.joint(nav.active).qpos[0])
                    observed = np.rad2deg(attempt.leaf.angle)
                    angles.setdefault(nav.active, []).append((attempt.source, abs(truth-observed)))
                if trace is not None:
                    resistance = (attempt.measured_force if nav.trial else attempt.force) if attempt else None
                    trace.write(json.dumps(dict(t=round(t, 3), phase=nav.phase, door=nav.active,
                                                position=p.tolist(), target=target.tolist(),
                                                source=attempt.source if attempt else None,
                                                angle_deg=np.rad2deg(attempt.leaf.angle) if attempt else None,
                                                force_n=resistance,
                                                measured_force_n=attempt.measured_force if attempt else None,
                                                output_limited=nav.trial.output_limited if nav.trial else False,
                                                force_target_n=nav.trial.target if nav.trial else None,
                                                trajectory_state=flight.mode if flight else None,
                                                velocity=pose.velocity[:2].tolist(), free_clearance_m=clearance,
                                                reference=reference[0].tolist() if flight and reference is not None else None,
                                                reference_speed_mps=float(np.linalg.norm(reference[1]))
                                                if flight and reference is not None else None))+"\n")
                if board:
                    status = {"tracking": "Minimum Snap", "validated_prefix": "继续已验证轨迹",
                              "braking": "安全制动", "holding": "暂停并重规划", "arrived": "到达轨迹端点",
                              "separating": "低速脱离门扇"}
                    board.update(t, p, target, nav, trajectory=flight.preview if flight else None,
                                 trajectory_status=status.get(flight.mode, "") if flight else "")
                if view:
                    for event in nav.events[previous_event:]:
                        if event["event"] not in ("remaining", "precontact_update"):
                            fields = {key: event[key] for key in ("t", "door", "event", "selected",
                                       "door_cost", "avoid_cost", "net", "reason", "force",
                                       "observed_load", "peak_force", "output_limited", "progress_deg")
                                      if key in event}
                            print(json.dumps(fields, ensure_ascii=False), flush=True)
                    previous_event = len(nav.events)
            position_gain = 1.15 if nav.phase == "fly" else 2.1
            velocity_target = position_gain * (target - pose.position[:2])
            speed = {"push": .36, "boost": .22, "approach": .62, "retreat": .52, "release": .82,
                     "cross": .82, "bypass": .72}.get(nav.phase, 1.28)
            if nav.phase == "release" and len(nav.path) >= 3:
                speed = .40
            if flight is not None and flight.mode == "separating" and nav.phase == "release":
                speed = min(speed, .58)
            velocity_target *= min(1., speed/max(np.linalg.norm(velocity_target), 1e-6))
            gain = 4.8 if nav.phase in ("push", "boost") else 3.4
            planar = gain*(velocity_target-pose.velocity[:2])
            flight_reference = flight.reference(t) if flight is not None else None
            if flight_reference is not None:
                p_ref, v_ref, a_ref, _ = flight_reference
                planar = a_ref + 4.8*(p_ref-pose.position[:2]) + 3.4*(v_ref-pose.velocity[:2])
            elif nav.trial is not None:
                tangent = nav.attempt.leaf.tangent
                baseline = float(planar @ tangent)
                planar += (nav.trial.acceleration-baseline)*tangent
            if t-unloading_started < UNLOAD_BLEND_S and nav.phase in ("release", "retreat"):
                fraction = (t-unloading_started)/UNLOAD_BLEND_S
                blend = fraction*fraction*(3.-2.*fraction)
                planar = (1.-blend)*contact_command + blend*planar
            motion = np.r_[np.clip(planar, -PUSH_ACCEL_LIMIT_MPS2, PUSH_ACCEL_LIMIT_MPS2),
                           np.clip(10*(ALTITUDE-pose.position[2])-5.8*pose.velocity[2], -4, 4)]
            thrust = rotor_control(pose.rotation, gyro, motion, sim.mass)
            step_peak = float(max(thrust))
            motor_peak = max(motor_peak, step_peak)
            motor_saturation_steps += int(step_peak >= 9.99)
            if nav.trial is not None and (
                    planar @ nav.attempt.leaf.tangent > motion[:2] @ nav.attempt.leaf.tangent + .03
                    or max(thrust) >= 9.99):
                limited_steps += 1
            sim.step(thrust)
            wall_contact = False
            passage_doors = set()
            for pair in sim.data.contact.geom[:sim.data.ncon]:
                first, second = map(int, pair)
                if first in watch_ids and sim.model.geom_group[second] == 1:
                    hit = first
                elif second in watch_ids and sim.model.geom_group[first] == 1:
                    hit = second
                else:
                    continue
                if hit in wall_ids:
                    wall_contact = True
                elif nav.phase in ("release", "cross"):
                    passage_doors.add(panel_names[hit])
            if wall_contact:
                touches["wall_steps"] += 1
            if passage_doors:
                touches["door_steps_during_passage"] += 1
                passage_contact.update(passage_doors)
                for name in passage_doors:
                    passage_contact_last[name] = round(t-passage_started, 3)
            if viewer and step % 10 == 0:
                viewer.cam.lookat[:] = [pose.position[0]+1.6, pose.position[1], .75]
                viewer.sync()
                elapsed = perf_counter()-frame_started
                sleep(max(0., 10*DT/REALTIME_FACTOR-elapsed))
                frame_started = perf_counter()
        result = dict(reached=reason == "goal", reason=reason, seconds=round(t, 3),
                      distance_m=round(float(distance), 3), seed=seed, noise_m=noise,
                      mode=mode, push_force_ramp_n_per_s=force_ramp,
                      wall_contact_steps=touches["wall_steps"],
                      passage_door_contact_steps=touches["door_steps_during_passage"],
                      passage_contact_by_door=dict(passage_contact),
                      last_passage_contact_after_switch_s=passage_contact_last,
                      optimize_trajectory=optimize_trajectory,
                      trajectory=flight.diagnostics() if flight else None,
                      tracking_p95_m=round(float(np.percentile(trajectory_errors, 95)), 4)
                      if trajectory_errors else None,
                      free_flight_min_clearance_m=round(free_clearance, 4) if np.isfinite(free_clearance) else None,
                      motor_peak_n=round(motor_peak, 3), motor_saturation_steps=motor_saturation_steps,
                      angle_audit={
                          name: dict(samples=len(rows),
                                     odometry_frames=sum(src == "contact odometry" for src, _ in rows),
                                     max_error_deg=round(max(error for _, error in rows), 2))
                          for name, rows in angles.items()},
                      events=nav.events)
        if trace is not None:
            trace.flush()
        if summary_file is not None:
            summary_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if board:
            board.update(t, pose.position[:2], target, nav, force=True)
            board.finish(result)
        if viewer and reason != "viewer_closed":
            print(f"{reason}: {result['seconds']:.2f} s, {result['distance_m']:.2f} m", flush=True)
            print("Close the MuJoCo window to exit.", flush=True)
            while viewer.is_running():
                viewer.sync()
                if board:
                    board.pump()
                sleep(.04)
        if board:
            board.close()
        return result


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR/"trace.jsonl").open("w", encoding="utf-8") as trace:
        run(view=SHOW_VIEWER, dashboard=SHOW_DASHBOARD, trace=trace,
            summary_file=OUTPUT_DIR/"result.json")


if __name__ == "__main__":
    main()
