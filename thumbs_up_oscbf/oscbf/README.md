# A7V2 MuJoCo OSCBF trajectory reproduction

This experiment replays the existing thumb-up trajectory on the combined A7V2 arm-hand URDF and inserts a spherical obstacle into the MuJoCo scene. A velocity-level Operational Space Control Barrier Function (OSCBF) filters the seven arm-joint velocity commands while the hand joints continue following the recorded gesture trajectory.

## What is reproduced

- Existing trajectory generator output is used as the upstream command.
- The A7V2 combined URDF is loaded directly by MuJoCo.
- Fourteen arm-attached collision spheres approximate whole-arm control geometry.
- Sphere-obstacle distance, arm joint limits, and velocity limits are assembled as CBF/QP constraints.
- The QP objective uses the A7 body Jacobian and null-space projection, following the task-consistent objective in the OSCBF paper.
- A nominal rollout and an OSCBF rollout are logged and compared.

This first implementation is the paper's kinematic/velocity-control formulation. It is not yet the torque-control HOCBF formulation: the source trajectory is a joint-position trajectory and the URDF-imported MuJoCo model has no configured actuators. The distinction is intentional and is reported explicitly rather than treating kinematic position writes as torque control.

> Packaging note: the bundled current trajectory lasts 6.7 s and has A4 = 0 rad.
> `outputs/oscbf_thumbsup/` preserves an earlier 7.7 s run with an invalid A4 reference.
> A new default run uses the current trajectory and will produce different values.
> Use `--output-dir` to preserve the archived first-experiment outputs.
> See [the project overview](../README.md) for the experiment/version map.

## Run headlessly

From this directory:

```powershell
python .\run_oscbf_trajectory.py
```

The default inputs are:

- URDF: `../models/a7v2_arm_hand/urdf/a7v2_latest_stand_single_arm_right_hand.urdf`
- trajectory: `../thumbsup/outputs/thumbs_up_trajectory.csv`

Outputs are written under `outputs/oscbf_thumbsup/`:

- `nominal_rollout.csv`
- `oscbf_rollout.csv`
- `summary.json`
- `trajectory_comparison.svg`

## Archived first-experiment result (7.7 s)

The archived output was generated with an earlier 7.7 s trajectory (1541
control steps over 7.7 s):

- The unfiltered baseline penetrates the expanded obstacle by 0.1334 m and is
  unsafe for 344 control steps.
- The OSCBF rollout has minimum obstacle barrier `h = 0.000121 m`; no sampled
  control step violates the obstacle or URDF joint-limit barriers.
- At 1.20 s, the OSCBF has moved A7 to `(0.599, -0.465, 0.684) m` instead of the
  nominal `(0.751, -0.399, 0.747) m`, thereby passing around the obstacle. By
  2.00 s it has rejoined the raised-arm portion of the trajectory.
- The pure-Python QP solver took 3.56 ms on average and 15.45 ms at the 95th
  percentile on the test machine. This is a functional prototype, not a claim
  that the current Python implementation meets the paper's control frequency.
- The largest arm-joint tracking error is 1.262 rad, dominated by avoidance and
  the incompatible A4 command described below; the final error is 0.138 rad.

## Run with the MuJoCo viewer

```powershell
python .\run_oscbf_trajectory.py --viewer
```

The safe motion repeats continuously until the viewer is closed. Use
`--repeat 3` to play exactly three cycles.

## Record an MP4

```powershell
python .\run_oscbf_trajectory.py `
  --record-video .\outputs\oscbf_thumbsup\oscbf_obstacle_avoidance.mp4
```

The default recording is 640x480 H.264 at 30 FPS. Use `--video-fps 60` or
`--video-repeats 2` to change the frame rate or repeat the motion in the file.

The solid dark-red sphere is the physical obstacle. The larger translucent
orange sphere shows `obstacle radius + safety margin`; the allowed region for
each blue robot collision sphere is outside this orange boundary. Blue
translucent sites are the arm collision spheres used by the CBF. Therefore,
when a blue sphere is just tangent to the orange boundary, `h = 0`. Both
obstacle geoms have no MuJoCo contact affinity, so avoidance is caused by the
OSCBF rather than by the physics contact solver.

## Important parameters

```powershell
python .\run_oscbf_trajectory.py `
  --obstacle 0.70 -0.315 0.82 `
  --obstacle-radius 0.105 `
  --safety-margin 0.025 `
  --alpha 10 `
  --dt 0.005
```

The safe condition for every arm sphere is

`h = distance - robot_radius - obstacle_radius - safety_margin >= 0`.

The first-order CBF constraint is

`grad(h) * qdot + alpha * h >= 0`.

OSCBF solves a seven-variable convex QP at every control step. The objective penalizes changes to the A7 operational-space motion and changes in the associated joint-space null space. Joint-position and joint-velocity constraints are included in the same QP.

The archived first-experiment trajectory commanded `A4_joint=-0.5 rad`, while the supplied URDF declares `A4_joint` range `[0, 2.2] rad`. The program does not silently clip this discrepancy. The nominal rollout exposes the upstream limit violation, while the OSCBF rollout treats the URDF limit as a safety constraint. `summary.json` reports obstacle and joint-limit violations separately.

## Scope and next step

This experiment uses known obstacle position (simulation ground truth), as in the original repository. A camera/point-cloud front end can later update obstacle position and radius without changing the CBF controller interface.

For a dynamic/torque-level reproduction, the next engineering step is to add torque actuators and stable servo gains to an MJCF model, then replace the first-order constraints with relative-degree-two HOCBF constraints using MuJoCo's mass matrix and bias forces.

## A4 URDF joint-limit comparison with the obstacle present

Run the controlled comparison with:

```powershell
python .\run_a4_limit_comparison.py
```

Both cases use the same obstacle, safety margin, OSCBF gains, camera, control
period, hand gesture, and all non-A4 upstream references.  The current source
trajectory keeps `A4_joint=0 rad`, which is valid for the URDF range
`[0, 2.2] rad`.  The comparison script creates a second trajectory by smoothly
injecting only the A4 reference down to `-0.5 rad` during the arm-raise motion.

The joint limits are read from MuJoCo's imported URDF `model.jnt_range` and are
included in the same QP as the obstacle CBFs:

`h_lower,i = q_i - q_i,min`, `qdot_i >= -alpha h_lower,i`

`h_upper,i = q_i,max - q_i`, `-qdot_i >= -alpha h_upper,i`

Outputs are written to `outputs/a4_joint_limit_comparison/`, including two
detailed CSV trajectories, two single-case videos, a synchronized side-by-side
video, barrier/intervention plots, and `comparison_summary.json`.

In the checked result, the violating upstream reference reaches `-0.5 rad`,
but the executed A4 minimum is approximately zero and no sampled joint-limit
or obstacle-barrier violation occurs.  OSCBF does not only clip A4: the
task-consistent metric redistributes motion across the other joints, so the
whole-arm and end-effector trajectories differ from the valid-upstream case.

## URDF joint-limit comparison

Run the controlled A4 limit-violation experiment with:

```powershell
python .\run_joint_limit_comparison.py
```

The script does not modify the source trajectory CSV. It constructs two
in-memory upstream references:

- valid case: the current A4 reference remains at `0 rad`, which is the URDF
  lower limit;
- invalid case: A4 smoothly ramps from `0 rad` to `-0.5 rad` during the arm
  raise and then holds the invalid command.

The obstacle is moved far outside the workspace in both cases so that the
experiment isolates the URDF joint-limit CBF. For every arm joint, OSCBF adds
the two constraints

`qdot_i >= -alpha * (q_i - q_i,min)`

`-qdot_i >= -alpha * (q_i,max - q_i)`.

Outputs are written to `outputs/joint_limit_comparison/`, including both
upstream references, both full rollouts, a JSON summary, two trajectory plots,
and a synchronized side-by-side MP4. The rollout CSV files log desired and
executed joint positions, nominal and filtered joint velocities, and the lower
and upper joint-limit barrier values for all seven arm joints.
