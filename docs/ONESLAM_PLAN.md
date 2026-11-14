# Stretch 3 adaptation of OneSLAM: plan

_Project plan (source: project notes, Nov 2026). Implementation status: see [ONESLAM.md](ONESLAM.md)._

You can build a Stretch 3 adaptation of OneSLAM using its existing cameras. I would make it an RGB-D system using the head camera, then add navigation around that. This is a substantial extension of OneSLAM rather than a camera-driver change.

I interpret “only cameras” as excluding LiDAR, wheel odometry, and IMU from localization. You will still need Stretch’s joint states to account for head movement.

Stretch 3 provides these useful camera roles: Stretch Docs
Camera	Role in your version
Head RealSense D435if RGB-D	Primary localization, metric mapping, and obstacle sensing
Head wide-angle RGB camera	Optional additional visual coverage; add after the first version works
Gripper RealSense D405	Close-range perception during therapy interactions or manipulation; not the primary room-navigation camera


Build the first version around the D435if alone. Its depth gives you metric geometry without adding hardware.
1. Establish a working camera-navigation baseline
Before modifying OneSLAM, run Stretch’s existing FUNMAP demo. It already builds maps from the head depth camera and uses depth for obstacle avoidance. This gives you a reference for what the hardware can accomplish. It does not establish a strictly camera-only localization baseline; audit its use of robot odometry separately. Stretch Docs

On a configured Stretch ROS 2 installation:
ros2 launch stretch_funmap mapping.launch.py

In another terminal:
ros2 run stretch_core keyboard_teleop --ros-args -p mapping_on:=True

Then collect recordings for your own pipeline. Discover actual topic names instead of assuming a namespace:
ros2 topic list -t

Record synchronized head RGB, depth aligned to RGB, camera calibration, /joint_states, /tf, and /tf_static. Keep original timestamps and depth units.
2. Reproduce unmodified OneSLAM on Stretch recordings
Fork OneSLAM, pin the upstream commit, and first run its original monocular pipeline on exported head RGB frames.

The upstream input is an image directory plus calibration, assuming a distortion-free pinhole camera. Rectify images before export, and update intrinsics for any resizing or rotation. Stretch’s head camera is mounted vertically; preserving the original optical orientation initially avoids unnecessary coordinate mistakes. github.com

Start with:
- Head held at a fixed pan/tilt angle.
- Arm stowed.
- An empty room with textured objects.
- Slow base translation and turns, including a return to the start.
This separates installation and calibration problems from your algorithm changes.
3. Add depth to the SLAM geometry
This is the main algorithmic contribution. Retain OneSLAM’s learned point tracking, but use measured depth to create metric landmarks.

For a tracked pixel \((u,v)\) with valid aligned depth \(z\):
\[
\mathbf p_C =
zK^{-1}
\begin{bmatrix}u\\v\\1\end{bmatrix}.
\]
Implement the following:
Component	Change
Dataset loader	Return RGB, aligned depth, intrinsics, timestamp, and camera frame
Landmark initialization	Back-project valid depth instead of assigning arbitrary initial depth
Pose estimation	Estimate camera pose using robust 3D-to-2D correspondences; optionally refine with 3D-to-3D alignment
Bundle adjustment	Add uncertainty-weighted depth constraints alongside reprojection residuals
Outlier rejection	Reject invalid depth, depth discontinuities, and inconsistent tracks


Depth initialization alone is insufficient: if subsequent optimization uses only monocular reprojection, metric scale is not reliably constrained. Keep depth observations in the optimization.

Likely integration points include run_slam.py, tracking.py, slam.py, optimizers.py, and the dataset/depth modules. Confirm the internal data flow before editing; these are proposed changes, not existing RGB-D features. The entry point already exposes tracking, depth-initialization, and bundle-adjustment settings. GitHub
4. Convert camera localization into base localization
Stretch’s camera moves when its head moves. Your navigation system must distinguish this from base motion.

Using \(^{A}T_B\) to mean “transform coordinates from frame \(B\) into frame \(A\)”:
\[
{}^{map}T_{base}(t)
=
{}^{map}T_{camera}(t)
\left({}^{base}T_{camera}(t)\right)^{-1}.
\]
Obtain the camera-to-base relationship from Stretch’s TF tree at the image timestamp.

First validate with the head fixed. Then test with the base stationary while moving only the head: the estimated base should remain stationary. This is a critical integration check.
5. Separate the static map from moving obstacles
For therapy environments, build two representations:
- Static localization map: stationary room geometry. Exclude people and robot-body pixels from landmark selection, with geometric checks for other moving objects.
- Live obstacle map: current depth observations, including patients, therapists, chairs, and equipment.
A person excluded from SLAM must remain visible to obstacle avoidance. Add self-filtering for Stretch’s arm and body.

Use a depth-derived occupancy or voxel map for navigation; OneSLAM’s sparse landmarks do not establish enough free space for collision checking. Nav2 supports obstacle representations derived from depth and point clouds. docs.nav2.org
6. Turn the offline pipeline into a ROS 2 component
Add a wrapper that publishes proposed interfaces such as:
/oneslam/camera_pose
/oneslam/base_pose
/oneslam/landmarks
/oneslam/tracking_status

Make these design requirements explicit:
- Bounded input queues and timestamped outputs.
- Measured capture-to-pose latency, including tracking-window delay.
- Tracking-loss detection and stopping when pose or depth becomes stale.
- A consistent map → odom → base_link arrangement with one publisher per transform.
- Relocalization and loop closure before attempting repeated, long-duration routes.
Profile recorded data before choosing onboard or external processing. Upstream documents GPU execution; its example timing does not demonstrate real-time operation on Stretch. github.com
7. Validate in stages before therapy use
Stage	Test
Metric reconstruction	Compare mapped distances against measured room distances
Base localization	Fixed-head routes, then head-only movement
Dynamic scenes	People crossing the view, occlusion, and moved furniture
Runtime	Pose age, latency percentiles, dropped frames, and tracking recovery
Navigation	Empty-room goals, narrow passages, obstacle appearance, and stopping
Therapy setting	Supervised trials after the previous stages pass


For a strict camera-only evaluation, ensure wheel odometry and IMU do not enter pose initialization, fusion, or fallback. You can log them separately for comparison, but they are not independent ground truth.

My recommended first milestone is a recorded-data RGB-D OneSLAM fork that produces a metric base trajectory with the head fixed. Once that works, tackle moving-head transforms, dynamic obstacles, and live navigation in that order. Camera field-of-view gaps will still constrain movement, especially reversing and turning near patients; plan sensing and stopping behavior around those gaps.