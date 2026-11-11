"""Microducks doing rounds in an ICU, simulated in Newton.

One ward is the SimReady ICU bay from ward.py plus three Microducks (Pollen Robotics' open robot) walking on Pollen's
RL-trained `velstand` policy. --worlds N copies the whole ward, ducks included, N times into one model, so every
copy steps together on one device.

    python rounds.py                          # 1 ward, 3 ducks, 14 simulated seconds -> out/rounds.npz + .json
    python rounds.py --worlds 256             # 256 wards (768 ducks): use an NVIDIA GPU
    python rounds.py --render out/rounds.npz  # replay a recording with Newton's OpenGL viewer -> out/rounds.mp4
"""
import argparse
import json
import math
import os
import time

import numpy as np
import onnxruntime as ort
import warp as wp

import newton
from ward import HERE, add_furniture, add_room

wp.config.log_level = getattr(wp, "LOG_WARNING", 30)  # hide Warp's per-kernel compile and load timings

DUCK_XML = os.path.join(HERE, "downloads", "microduck", "robot_groundcontact.xml")
POLICY = os.path.join(HERE, "downloads", "policy", "velstand_batched.onnx")  # velstand.onnx with a dynamic batch

SIM_DT, SUBSTEPS = 0.002, 10  # physics at 500 Hz, policy at 50 Hz, as the policy was trained
CTRL_DT = SIM_DT * SUBSTEPS
UPRIGHT_Z = 0.08  # a duck counts as upright while its trunk is above 8 cm (it stands at about 12.5 cm)

# Hero ward: three ducks walk in from the open side to the patients. (start x, y, heading), (goal x, y, heading),
# start delay in seconds.
HERO = [
    ((-0.35, -1.25, 90.0), (-1.05, 0.95, 180.0), 0.4),  # left bedside, facing the patient
    ((0.35, -1.05, 90.0), (1.8, -0.05, 90.0), 0.9),  # foot of the right bed, inside the curtain
    ((0.0, -1.85, 90.0), (-0.15, 0.35, 180.0), 1.4),  # the aisle between the beds
]
# Every other ward sends its ducks from random starts to random spots from this list.
TARGETS = [(-1.05, 0.95, 180.0), (1.8, -0.05, 90.0), (-0.15, 0.35, 180.0), (1.05, -1.2, 0.0), (-1.9, -1.0, 180.0),
           (-2.3, 0.2, 90.0)]

# Camera for --render: a slow push-in on the hero ducks, then a pull-back that reveals every ward.
CAMERA = dict(fov=52.0, near_target=(0.0, -0.15, 0.28), near_dist=3.0, near_pitch=-14.0, near_yaw=125.0,
              far_pitch=-56.0, far_yaw=78.0, reveal_start=7.0, reveal_end=12.5, spacing=8.0)


def load_policy():
    """The ONNX policy carries its joint order and default pose as metadata."""
    sess = ort.InferenceSession(POLICY)
    meta = sess.get_modelmeta().custom_metadata_map
    names = meta["joint_names"].split(",")
    default = np.array([float(x) for x in meta["default_joint_pos"].split(",")], np.float32)
    return sess, names, default


def build_model(worlds, ducks, names, default):
    """One ward (ducks + room + furniture), replicated `worlds` times into a single Newton model."""
    ward = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(ward)
    agents = []
    for _ in range(ducks):
        j0 = ward.joint_count
        ward.add_mjcf(DUCK_XML, xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
        by_name = {ward.joint_label[j].split("/")[-1]: j for j in range(j0, ward.joint_count)}
        agents.append({"q0": ward.joint_q_start[j0], "qd0": ward.joint_qd_start[j0],  # the free joint
                       "idx_q": [ward.joint_q_start[by_name[n]] for n in names],
                       "idx_qd": [ward.joint_qd_start[by_name[n]] for n in names]})
        for i, v in zip(agents[-1]["idx_q"], default):
            ward.joint_q[i] = float(v)
    robot_bodies = ward.body_count
    s0 = ward.shape_count
    add_room(ward)
    placed = add_furniture(ward)
    # Room and furniture share one negative collision group: they never collide with each other, only with ducks.
    for i in range(s0, ward.shape_count):
        if ward.shape_collision_group[i] != 0:
            ward.shape_collision_group[i] = -2
    coords, dofs, bodies = ward.joint_coord_count, ward.joint_dof_count, ward.body_count

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(ward, worlds)
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))

    # Flat indices of every duck's free joint and policy joints across all worlds.
    q0 = np.array([w * coords + a["q0"] for w in range(worlds) for a in agents], np.int64)
    qd0 = np.array([w * dofs + a["qd0"] for w in range(worlds) for a in agents], np.int64)
    idx_q = np.array([[w * coords + i for i in a["idx_q"]] for w in range(worlds) for a in agents], np.int64)
    idx_qd = np.array([[w * dofs + i for i in a["idx_qd"]] for w in range(worlds) for a in agents], np.int64)
    robot_idx = np.concatenate([np.arange(w * bodies, w * bodies + robot_bodies) for w in range(worlds)])
    return model, placed, robot_idx, q0, qd0, idx_q, idx_qd


def hero_world(worlds):
    """The ward in the middle of the square grid the viewer lays out."""
    side = int(math.ceil(math.sqrt(worlds)))
    return min(worlds - 1, (side // 2) * side + side // 2)


def make_plan(worlds, ducks, hero, seed=11):
    rng = np.random.default_rng(seed)
    plan = []
    for w in range(worlds):
        if w == hero:
            plan += HERO[:ducks]
            continue
        picks = rng.choice(len(TARGETS), size=ducks, replace=False)
        for j in range(ducks):
            start = (float(rng.uniform(-1.0, 1.0)) + (j - 1) * 0.5, float(rng.uniform(-2.2, -0.9)),
                     float(rng.uniform(60, 120)))
            plan.append((start, TARGETS[int(picks[j])], float(rng.uniform(0.2, 2.0))))
    return plan


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def smooth(a, b, t):
    t = max(0.0, min(1.0, t))
    t = t * t * (3 - 2 * t)
    return a + (b - a) * t


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def commands(jq, q0, plan, t):
    """Where to walk. The RL policy does the walking; this waypoint controller only turns each duck's goal into the
    policy's velocity command (forward speed in slot 0, yaw rate in slot 2 of its 13-D command)."""
    cmd = np.zeros((len(q0), 13), np.float32)
    for a, qs in enumerate(q0):
        _, (tx, ty, th), delay = plan[a]
        if t < delay:
            continue
        x, y = jq[qs], jq[qs + 1]
        yaw = yaw_of(jq[qs + 3:qs + 7])
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist > 0.12:
            err = wrap(math.atan2(dy, dx) - yaw)
            cmd[a, 0] = 0.5 * max(0.0, math.cos(err)) ** 3 * min(1.0, dist / 0.3)
            cmd[a, 2] = max(-1.0, min(1.0, 1.6 * err))
        else:
            err = wrap(math.radians(th) - yaw)
            if abs(err) > 0.15:
                cmd[a, 2] = max(-0.8, min(0.8, 1.6 * err))
    return cmd


def simulate(args):
    t0 = time.time()
    device = wp.get_device()
    sess, names, default = load_policy()
    model, placed, robot_idx, q0, qd0, idx_q, idx_qd = build_model(args.worlds, args.ducks, names, default)
    n = len(q0)
    print(f"[build] device={device.alias} ({device.name}) worlds={args.worlds} ducks={n} bodies={model.body_count} "
          f"shapes={model.shape_count} build_s={time.time() - t0:.1f}", flush=True)
    solver = newton.solvers.SolverMuJoCo(model, solver="newton", nconmax=args.nconmax, njmax=args.njmax)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    hero = hero_world(args.worlds)
    plan = make_plan(args.worlds, args.ducks, hero)

    jq = s0.joint_q.numpy()
    for a, qs in enumerate(q0):
        (x, y, heading), _, _ = plan[a]
        jq[qs:qs + 3] = [x, y, 0.125]
        jq[qs + 3:qs + 7] = [0.0, 0.0, math.sin(math.radians(heading) / 2), math.cos(math.radians(heading) / 2)]
    s0.joint_q.assign(jq)
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    ctrl.joint_target_q = wp.clone(s0.joint_q)
    target_q = ctrl.joint_target_q.numpy()

    last_action = np.zeros((n, len(names)), np.float32)
    gravity = np.array([0.0, 0.0, -1.0])
    steps = int(args.seconds / CTRL_DT)
    every = max(1, int(round(1.0 / (args.fps * CTRL_DT))))
    frames, log = [], []
    t_loop, t = time.time(), 0.0
    for k in range(steps):
        jq, jqd = s0.joint_q.numpy(), s0.joint_qd.numpy()
        # The 61-D observation the policy was trained on.
        obs = np.zeros((n, 61), np.float32)
        for a in range(n):
            R = rot(jq[q0[a] + 3:q0[a] + 7])
            obs[a, 0:3] = R.T @ jqd[qd0[a] + 3:qd0[a] + 6]  # angular velocity, body frame
            obs[a, 3:6] = R.T @ gravity  # gravity direction, body frame
        obs[:, 6:20] = jq[idx_q] - default  # joint positions relative to the default pose
        obs[:, 20:34] = jqd[idx_qd]  # joint velocities
        obs[:, 34:48] = last_action
        obs[:, 48:61] = commands(jq, q0, plan, t)
        action = sess.run(None, {"obs": obs})[0].astype(np.float32)  # one batched call for every duck
        last_action = action
        target_q[idx_q] = default + action  # position targets for the MJCF's position actuators
        ctrl.joint_target_q.assign(target_q)
        for _ in range(SUBSTEPS):
            s0.clear_forces()
            solver.step(s0, s1, ctrl, None, SIM_DT)
            s0, s1 = s1, s0
        t += CTRL_DT
        if k % every == 0:
            frames.append(s0.body_q.numpy()[robot_idx].astype(np.float32))
        if k % 50 == 0:
            z = jq[q0 + 2]
            log.append({"t": round(t, 2), "upright": int((z > UPRIGHT_Z).sum()), "of": n})
            print("[sim]", log[-1], f"wall_s={time.time() - t0:.1f}", flush=True)

    z = s0.joint_q.numpy()[q0 + 2]
    summary = {"robot": "microduck", "worlds": args.worlds, "ducks_per_ward": args.ducks, "ducks": n,
               "device": f"{device.alias} ({device.name})", "sim_seconds": args.seconds,
               "loop_wall_s": round(time.time() - t_loop, 1), "upright_at_end": int((z > UPRIGHT_Z).sum()),
               "hero": hero, "frames": len(frames), "fps": 1.0 / (every * CTRL_DT), "bodies": model.body_count,
               "shapes": model.shape_count, "placed": placed, "log": log}
    os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
    np.savez_compressed(args.record, body_q=np.stack(frames), robot_idx=robot_idx, fps=summary["fps"],
                        meta=json.dumps({k: v for k, v in summary.items() if k not in ("placed", "log")}))
    with open(os.path.splitext(args.record)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("[done]", json.dumps({k: v for k, v in summary.items() if k not in ("placed", "log")}), flush=True)


def render(args):
    """Rebuild the same model, write the recorded duck poses into it frame by frame, and draw it with ViewerGL."""
    import imageio.v2 as imageio

    t0 = time.time()
    rec = np.load(args.render)
    body_q, robot_idx, fps = rec["body_q"], rec["robot_idx"], float(rec["fps"])
    meta = json.loads(str(rec["meta"]))
    worlds, ducks = int(meta["worlds"]), int(meta["ducks_per_ward"])
    _, names, default = load_policy()
    model, _, idx2, *_ = build_model(worlds, ducks, names, default)
    assert np.array_equal(idx2, robot_idx), "this recording was made with a different scene"
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    viewer.set_model(model)
    cam = CAMERA
    if worlds > 1:
        viewer.set_world_offsets((cam["spacing"], cam["spacing"], 0.0))
    offsets = viewer.world_offsets.numpy() if worlds > 1 else np.zeros((1, 3))
    hero, center = offsets[int(meta["hero"])], offsets.mean(0)
    side = cam["spacing"] * math.ceil(math.sqrt(worlds))
    viewer.camera.fov = cam["fov"]

    stem = os.path.splitext(args.render)[0]
    if args.still_at is not None:
        frame_ids = [min(len(body_q) - 1, int(args.still_at * fps))]
        out = f"{stem}_t{args.still_at:g}.png"
        writer = None
    else:
        frame_ids = range(len(body_q))
        out = stem + ".mp4"
        writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                    macro_block_size=2)
    bq = state.body_q.numpy()
    for count, fi in enumerate(frame_ids):
        t = fi / fps
        bq[robot_idx] = body_q[fi]
        state.body_q.assign(bq)
        u = (t - cam["reveal_start"]) / (cam["reveal_end"] - cam["reveal_start"])
        us = smooth(0.0, 1.0, u)
        push = min(1.0, max(0.0, t / cam["reveal_start"]))
        tx, ty, tz = cam["near_target"]
        target = np.array([smooth(hero[0] + tx, center[0], u), smooth(hero[1] + ty, center[1], u), smooth(tz, 0.0, u)])
        d_near = cam["near_dist"] - 0.35 * push
        dist = d_near * ((1.18 * side) / d_near) ** us  # log-scale pull-back to the whole grid
        rise = smooth(0.0, 1.0, min(1.0, 1.8 * max(0.0, u)))  # pitch steepens first so the camera clears the walls
        yaw = cam["near_yaw"] + (cam["far_yaw"] - cam["near_yaw"]) * us
        pitch = cam["near_pitch"] + (cam["far_pitch"] - cam["near_pitch"]) * rise
        cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        look = np.array([cp * math.cos(math.radians(yaw)), cp * math.sin(math.radians(yaw)), sp])
        viewer.set_camera(wp.vec3(*(target - dist * look).tolist()), pitch=pitch, yaw=yaw)
        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.end_frame()
        image = viewer.get_frame().numpy()
        if writer is None:
            imageio.imwrite(out, image)
        else:
            writer.append_data(image)
        if count % 50 == 0:
            print(f"[render] frame {count}/{len(frame_ids)} t={t:.2f} wall_s={time.time() - t0:.1f}", flush=True)
    if writer is not None:
        writer.close()
    print("[done] wrote", out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worlds", type=int, default=1, help="copies of the ward to simulate in parallel")
    p.add_argument("--ducks", type=int, default=3, choices=[1, 2, 3], help="Microducks per ward")
    p.add_argument("--seconds", type=float, default=14.0, help="simulated seconds")
    p.add_argument("--fps", type=int, default=25, help="recorded frames per simulated second")
    p.add_argument("--record", default=os.path.join(HERE, "out", "rounds.npz"), help="where to save the recording")
    p.add_argument("--render", metavar="NPZ", help="render this recording instead of simulating")
    p.add_argument("--still-at", type=float, help="with --render: write one PNG at this time (s) instead of a video")
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1350)
    p.add_argument("--nconmax", type=int, default=400, help="contact buffer size per world")
    p.add_argument("--njmax", type=int, default=1200, help="constraint buffer size per world")
    p.add_argument("--device", help='Warp device, e.g. "cuda:0" or "cpu" (default: the best available)')
    args = p.parse_args()
    if args.device:
        wp.set_device(args.device)
    if args.render:
        render(args)
    else:
        simulate(args)


if __name__ == "__main__":
    main()
