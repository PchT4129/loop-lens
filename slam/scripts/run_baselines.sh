set -e
SEQ=/home/yixiao/datasets/TUM/rgbd_dataset_freiburg3_long_office_household
ASSOC=/home/yixiao/ORB_SLAM3/Examples/RGB-D/associations/fr3_office.txt
VOC=/home/yixiao/ORB_SLAM3/Vocabulary/ORBvoc.txt
EXE=/home/yixiao/ORB_SLAM3/Examples/RGB-D/rgbd_tum
CFG=/home/yixiao/dev/vpr-loop-closure/slam/configs
for mode in loop_off loop_on; do
  cd /home/yixiao/dev/vpr-loop-closure/slam/runs/$mode
  echo "=== $mode ==="
  $EXE $VOC $CFG/TUM3_$mode.yaml $SEQ $ASSOC
done
