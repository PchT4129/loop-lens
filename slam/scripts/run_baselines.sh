set -e
SEQ=$HOME/datasets/TUM/rgbd_dataset_freiburg3_long_office_household
ASSOC=$HOME/ORB_SLAM3/Examples/RGB-D/associations/fr3_office.txt
VOC=$HOME/ORB_SLAM3/Vocabulary/ORBvoc.txt
EXE=$HOME/ORB_SLAM3/Examples/RGB-D/rgbd_tum
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CFG=$REPO/slam/configs
for mode in loop_off loop_on; do
  cd "$REPO/slam/runs/$mode"
  echo "=== $mode ==="
  $EXE $VOC $CFG/TUM3_$mode.yaml $SEQ $ASSOC
done
