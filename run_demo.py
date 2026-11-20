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

from door_resistance import ThrustResidual
from localization import Localization
from planner import Navigator, PUSH_ACCEL_LIMIT_MPS2
from scene import make_scene
from simulator import ALTITUDE, DT, RAYS, Simulator, rotor_control
from viewer_pause import ViewerPause

SHOW_VIEWER = True
SHOW_DASHBOARD = True
REALTIME_FACTOR = 1.0
EFFORT_WEIGHT_M_PER_J = 1/.030
PUSH_FORCE_RAMP_N_PER_S = 1.20
SEED = 7
RANGE_NOISE_M = .003
SCAN_STEPS = 25
OUTPUT_DIR = Path(__file__).with_name("runs")
MODE = "auto"


def run(*, view=False, dashboard=False, check=False, seed=SEED,
        noise=RANGE_NOISE_M, mode=MODE, trace=None, summary_file=None,
        force_ramp=PUSH_FORCE_RAMP_N_PER_S):
    arena, fixtures, names = make_scene()
    sim = Simulator(arena, fixtures, seed, noise)
    pose = Localization(arena, ALTITUDE)
    nav = Navigator(arena, effort_weight=EFFORT_WEIGHT_M_PER_J, check=check, mode=mode,
                    show_routes=view and dashboard, force_ramp=force_ramp)
    load = ThrustResidual(sim.mass)
    panel_ids = {name: sim.model.geom(name+"_panel").id for name in names}
    panel_names = {index: name for name, index in panel_ids.items()}
    wall_ids = {sim.model.geom(f"wall_{index}").id for index in range(len(arena.walls))}
    watch_ids = set(panel_ids.values()) | wall_ids
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
                pose.predict(acc, gyro, DT)
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
                target = nav.decide(p, pose.velocity[:2], points, reaction, t,
                                    planar_command=motion[:2],
                                    output_limited=limited_steps >= SCAN_STEPS//2)
                limited_steps = 0
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
                                                force_target_n=nav.trial.target if nav.trial else None))+"\n")
                if board:
                    board.update(t, p, target, nav)
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
            velocity_target *= min(1., speed/max(np.linalg.norm(velocity_target), 1e-6))
            gain = 4.8 if nav.phase in ("push", "boost") else 3.4
            planar = gain*(velocity_target-pose.velocity[:2])
            if nav.trial is not None:
                tangent = nav.attempt.leaf.tangent
                baseline = float(planar @ tangent)
                planar += (nav.trial.acceleration-baseline)*tangent
            motion = np.r_[np.clip(planar, -PUSH_ACCEL_LIMIT_MPS2, PUSH_ACCEL_LIMIT_MPS2),
                           np.clip(10*(ALTITUDE-pose.position[2])-5.8*pose.velocity[2], -4, 4)]
            thrust = rotor_control(pose.rotation, gyro, motion, sim.mass)
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
