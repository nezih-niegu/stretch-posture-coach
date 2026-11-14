#!/usr/bin/env bash
# Run OneSLAM (https://github.com/arcadelab/OneSLAM) on a session exported with
#   posture-coach scene export-oneslam <session>
# Run this on a Linux machine with an NVIDIA GPU (~5 GB VRAM), conda and cmake - NOT on the Stretch (no CUDA GPU).
#
#   ./scripts/run_oneslam.sh ./stretch_20261114_153512 [extra run_slam.py arguments]
#
# OneSLAM is cloned at a pinned commit into $ONESLAM_DIR (default ~/OneSLAM). It has no license file, so this repo
# does not include or modify its code; it only prepares the input and reads its output (poses_pred.txt).
set -euo pipefail
[[ $# -ge 1 ]] || { sed -n '2,9p' "$0"; exit 1; }
DATA="$(cd "$1" && pwd)"; shift
ONESLAM_DIR="${ONESLAM_DIR:-$HOME/OneSLAM}"
COMMIT="05c1543d79ecb7b1567bab31adc85a16a52c9c2a"

[[ -f "$DATA/calibration.json" && -d "$DATA/images" ]] || { echo "$DATA is not a OneSLAM input folder"; exit 1; }
python3 - "$DATA" <<'PY'
import json, sys
n = json.load(open(sys.argv[1] + "/stretch_export.json"))
print(f"{n['frames']} frames @ {n['fps']} Hz, camera travel {n['camera_travel_m']} m")
if n.get("warning"): print("WARNING:", n["warning"])
PY

if [[ ! -d "$ONESLAM_DIR/.git" ]]; then
  git clone https://github.com/arcadelab/OneSLAM "$ONESLAM_DIR"
fi
cd "$ONESLAM_DIR"
git fetch -q origin || true
git checkout -q "$COMMIT"
if [[ ! -d trained_models/cotracker ]]; then
  ./init_submodules.sh
  ./get_tap_model.sh
fi
command -v conda >/dev/null || { echo "conda is required (Miniconda/Miniforge)"; exit 1; }
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx OneSLAM; then
  source ./install.sh   # creates the OneSLAM env and builds g2opy (needs cmake)
fi
conda activate OneSLAM

DEV="cuda:0"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "WARNING: no CUDA GPU found - running on CPU will be very slow"; DEV="cpu"; }
SUGGEST="$(python -c "import json; print(json.load(open('$DATA/stretch_export.json'))['suggested_args'])")"
NAME="stretch_$(date +%Y%m%d_%H%M%S)"
python run_slam.py --data_root "$DATA" --name "$NAME" --target_device "$DEV" $SUGGEST "$@"

OUT="$ONESLAM_DIR/experiments/$NAME/poses_pred.txt"
echo
echo "Done: $OUT"
echo "Copy it to the robot and import it:"
echo "  scp $OUT hello-robot@<robot-ip>:/tmp/poses_pred.txt"
echo "  posture-coach scene import-oneslam <session> /tmp/poses_pred.txt"
