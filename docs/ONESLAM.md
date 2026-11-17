# Scene understanding and OneSLAM

Every session records **RGB-D keyframes** from the head D435i (color, aligned depth, a person mask) together with
the **head joint states** (pan/tilt) and **base odometry**, in `sessions/<id>/scene/`. The session report turns
them into a *Scene understanding* section:

* a **static 3-D map** of the room (5 cm voxels). People are removed using MediaPipe masks plus a carved column
  around every tracked person position.
* a **top-down map**: free floor, obstacles colored by height, unseen areas, Stretch's path, and an orange heat
  map of where the person spent time.
* **statistics**: observed / free floor area, obstacle footprint, mapped extent, the person's distance walked,
  floor area used, **free space around them** (distance to the nearest obstacle), distance to Stretch, and
  posture samples.
* a **3-D viewer** (`/sessions/<id>/scene3d`) and a **PLY point cloud** download (MeshLab, CloudCompare, Blender).

## Two pose sources

| | kinematic (default, on the robot) | OneSLAM (camera-only, offline on a GPU) |
|---|---|---|
| Camera pose from | head pan/tilt + wheel odometry | OneSLAM visual tracking (CoTracker + BA) |
| Metric scale | yes (joint geometry) | recovered from the D435i depth (camera-only) |
| Works when only the head moves | yes | no (monocular SLAM needs translation) |
| Use for | everyday reports | camera-only localization research, validation |

The kinematic poses are a **reference, not ground truth**: they use wheel odometry when the base moves.

## OneSLAM workflow

[OneSLAM](https://github.com/arcadelab/OneSLAM) is a monocular SLAM pipeline (CoTracker point tracking + g2o
bundle adjustment). Upstream needs an NVIDIA GPU: its minimal example takes about 7 minutes on a Quadro RTX 6000
and about 5 GB of VRAM. The Stretch's NUC has no CUDA GPU, so it runs **offline on another machine**. The
repository has **no license file**, so this project does not include or modify its code. `scripts/run_oneslam.sh`
clones a **pinned commit** (`05c1543`) on the GPU machine.

1. **Record a suitable session.** Upstream is monocular, so the camera must *translate*: drive the base slowly
   (or use `posture-coach follow`). Keep the head fixed and record more keyframes:

   ```bash
   posture-coach run -s robot.head_tracking=false -s scene.record_hz=10
   ```

   Start in an empty room with textured objects, then turn slowly and return to the start (plan step 2).

2. **Export** (on the robot):

   ```bash
   posture-coach scene export-oneslam <session>
   ```

   This writes `sessions/<id>/scene/oneslam_input/` in OneSLAM's format:
   * `images/00000000.jpg` …, the upright head image (the camera is mounted sideways; the intrinsics are
     rotated to match)
   * `calibration.json` with `{"intrinsics": {"fx","fy","cx","cy"}, "fps"}` (pinhole, no distortion)
   * `depth_mm/` aligned 16-bit depth in mm, at image resolution. `--with-depths` also writes `depths/`, the
     folder OneSLAM's loader reads, for an RGB-D fork.
   * `poses_kinematic.txt` (TUM, camera-to-map). Deliberately **not** named `poses_gt.txt`: it is a reference, not
     ground truth.
   * `frame_index.jsonl` (joint states per frame) and `stretch_export.json` (suggested `--depth_scale`, a warning
     if the camera barely translated)

   It was checked against OneSLAM's own `datasets.ImageDataset`: frames, intrinsics and depth load correctly.

3. **Run OneSLAM** on the GPU machine (needs conda + cmake):

   ```bash
   scp -r hello-robot@<robot-ip>:~/Projects/stretch-posture-coach/sessions/<id>/scene/oneslam_input ./stretch_<id>
   ./scripts/run_oneslam.sh ./stretch_<id>
   ```

   The first time, this clones OneSLAM, fetches the CoTracker weights and builds the `OneSLAM` conda env. It runs
   `run_slam.py --data_root ... --depth_scale <median scene depth>` and prints the path of `poses_pred.txt`.

4. **Import** (on the robot):

   ```bash
   posture-coach scene import-oneslam <session> /tmp/poses_pred.txt
   ```

   * **Metric scale (camera-only).** For keyframe pairs, the depth of frame *j* is moved into frame *i* with
     OneSLAM's relative pose and translation `s·t`, and `s` minimizes the median relative depth error against
     frame *i*'s measured depth. Only pairs with a clear minimum count (pure rotation gives none). The report
     shows the scale, the number of pairs and their spread.
   * **Map frame.** The trajectory is anchored at the first keyframe's kinematic camera pose, which gives the
     gravity direction and camera height from the head tilt joint.
   * **Base poses (plan step 4).** `T_map_base(t) = T_map_cam(t) · T_base_cam(pan(t), tilt(t))⁻¹`
   * **Validation (plan step 7)** appears in the report: camera path vs. the kinematic reference (RMSE and final
     error), and the **head-only check**. In segments where the wheels did not move but the head did, the
     OneSLAM base pose must stay still (drift < 10 cm → pass).
   * The scene map is rebuilt with OneSLAM poses. Frames OneSLAM did not localize (non-keyframes) use the
     kinematic pose.

## Status against the plan ([ONESLAM_PLAN.md](ONESLAM_PLAN.md))

| Plan step | Status |
|---|---|
| 1. Camera-navigation baseline (FUNMAP) | not part of this project (ROS 2). Use FUNMAP as the reference. |
| 2. Unmodified OneSLAM on Stretch recordings | **done**: export in upstream format, pinned upstream run script, import |
| 3. Depth in the SLAM geometry | **partial**: measured depth recovers the metric scale *after* SLAM, and `--with-depths` exports `depths/` for a fork. Depth in landmark initialization and bundle adjustment (the main algorithmic contribution) needs a OneSLAM fork: upstream currently initializes with a constant depth (`DepthEstimatorConstant`). |
| 4. Camera → base localization | **done**: head joint states, head-only stationarity check |
| 5. Static map vs. moving obstacles | **done for mapping**: people masked and carved from the static map, tracked separately. Navigation costmaps: not implemented. |
| 6. ROS 2 component | not implemented (offline pipeline) |
| 7. Staged validation | **partial**: metric map statistics, head-only check, drift vs. kinematic reference. Runtime/latency and navigation trials: not applicable offline. |

## Limitations

* The head kinematics are approximate: pan axis at the base origin, camera 5 cm ahead of it, at
  `camera.height_m`. Expect a few cm of error. Tune `scene.pan_axis_x/y` and `scene.camera_forward_m`.
* A person who stays still for the whole session is indistinguishable from furniture only where the detector
  missed them. Everywhere they were detected, they are removed.
* Furniture within `scene.person_radius_m` (35 cm) of where someone stood is removed together with them.

## Update: live registration as the default reference

Sessions now record a **registered camera pose with every keyframe**:
* 2-D lidar SLAM gives the base pose,
* the head joint states give the camera direction,
* the floor plane gives tilt, roll and height.

`scene build` uses these poses by default (`source: registered-lidar` in `scene.json`). They are a much
stronger reference than wheel odometry for judging OneSLAM's camera-only trajectory.

When you import OneSLAM poses, the report uses them (`oneslam+registered-lidar`). It falls back to the
registered poses for frames OneSLAM didn't localize.

Older sessions without registered poses are rebuilt with the floor-plane correction (`kinematic+floor`).
