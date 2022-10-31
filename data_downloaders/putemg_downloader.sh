#!/bin/bash

BASE_URL="https://chmura.put.poznan.pl/s/G285gnQVuCnfQAx/download?path=%2F"
DIR_BASE="${PWD}/datasets/putemg/raw"
DIR_CSV="${DIR_BASE}/data_csv"
DIR_HDF5="${DIR_BASE}/data_hdf5"
DIR_DEPTH="${DIR_BASE}/depth"
DIR_VIDEO_1080P="${DIR_BASE}/video_1080p"
DIR_VIDEO_576P="${DIR_BASE}/video_576p"

# wget download command with the first argument is to continue an interrupted download where it was stopped and
# the second one is to check the timestamps to prevent the download of the same file
WGET="wget -c -N"

# allow the terminal to throw an exception where it finds the error and then the code stop execution
set -e

function usage() {
  echo 'Usage:' $(basename "$0") '<experiment_types> <media_types> [<id1> <id2> ..]'
  echo
  echo 'Arguments:'
  echo '  <experiment_types>   comma-separate list of experiment types (supported: emg_gesture, emg_force)'
  echo '  <media_types>        comma-separate list of media (supported: data_csv, data_hdf5, depth, video_1080p, video_576p)'
  echo
  echo 'Examples:'
  echo $(basename "$0") emg_gesture data_hdf5,video_1080p
  echo $(basename "$0") emg_gesture,emg_force data_csv,depth 03 04 07
}

if [ "$#" -lt 2 ]; then
  echo "Illegal number of parameters"
  usage
  exit 1
fi

# parse experiment types
if [ "$1" == "emg_gesture" ]; then
  EXPERIMENT_TYPES="emg_gestures"
elif [ "$1" == "emg_force" ]; then
  EXPERIMENT_TYPES="emg_force"
elif [ "$1" == "emg_gesture,emg_force" ] || [ "$1" == "emg_force,emg_gesture" ]; then
  EXPERIMENT_TYPES="(emg_gestures|emg_force)"
else
    echo "Invalid experiment type $1"
    usage
    exit 1
fi

# parse media types
IFS=',' read -r -a MEDIA_TYPES <<< "$2"
DATA_CSV=0
DATA_HDF5=0
DATA_DEPTH=0
DATA_VIDEO_1080P=0
DATA_VIDEO_576P=0

for m in "${MEDIA_TYPES[@]}"; do
  case "$m" in
  "data_csv")
    DATA_CSV=1
    ;;
  "data_hdf5")
    DATA_HDF5=1
    ;;
  "depth")
    DATA_DEPTH=1
    ;;
  "video_1080p")
    DATA_VIDEO_1080P=1
    ;;
  "video_576p")
    DATA_VIDEO_576P=1
    ;;
  *)
    echo "Invalid media type $m"
    usage
    exit 1
    ;;
  esac
done

# parse ids
if [ "$#" -gt 2 ]; then
  shift;shift
  IDS='('
  for id in "$@"; do
    if [[ $id =~ ^[0-9]{2}$ ]]; then
      IDS=${IDS}$id\|
    else
      echo "Invalid ID $id"
      usage
      exit 1
    fi
  done
  IDS=${IDS%"|"}\)
else
  IDS="[0-9]{2}"
fi

echo "$IDS"

echo EXPERIMENT_TYPES: $EXPERIMENT_TYPES
echo DATA_CSV: $DATA_CSV
echo DATA_HDF5: $DATA_HDF5
echo DATA_DEPTH: $DATA_DEPTH
echo DATA_VIDEO_1080P: $DATA_VIDEO_1080P
echo DATA_VIDEO_576P: $DATA_VIDEO_576P

REGEX="${EXPERIMENT_TYPES}-${IDS}"

echo $REGEX

records=$($WGET "${BASE_URL}"'&files=records.txt' -O - --quiet | grep -E "$REGEX")

# create directories
if [ $DATA_CSV -eq 1 ]; then
  mkdir -p "${DIR_CSV}"
fi
if [ $DATA_HDF5 -eq 1 ]; then
  mkdir -p "${DIR_HDF5}"
fi
if [ $DATA_DEPTH -eq 1 ]; then
  mkdir -p "${DIR_DEPTH}"
fi
if [ $DATA_VIDEO_1080P -eq 1 ]; then
  mkdir -p "${DIR_VIDEO_1080P}"
fi
if [ $DATA_VIDEO_576P -eq 1 ]; then
  mkdir -p "${DIR_VIDEO_576P}"
fi

echo "### Downloading putEMG..."

for r in $records; do
  if [ $DATA_CSV -eq 1 ]; then
    $WGET "${BASE_URL}DATA-CSV&files=${r}.zip" -O "${DIR_CSV}/${r}.zip"
    unzip -q "${DIR_CSV}/${r}.zip" -d "${DIR_CSV}/${r}"
    rm -f "${DIR_CSV}/s${r}.zip"
  fi
  if [ $DATA_HDF5 -eq 1 ]; then
    $WGET "${BASE_URL}Data-HDF5&files=${r}.hdf5" -O "${DIR_HDF5}/${r}.hdf5"
  fi
  if [ $DATA_DEPTH -eq 1 ]; then
    if [[ "${r}" =~ ^emg_gesture.* ]]; then
      $WGET "${BASE_URL}Depth&files=${r}.zip" -O "${DIR_DEPTH}/${r}.zip"
      unzip -q "${DIR_DEPTH}/${r}.zip" -d "${DIR_DEPTH}/${r}"
      rm -f "${DIR_DEPTH}/s${r}.zip"
    fi
  fi
  if [ $DATA_VIDEO_1080P -eq 1 ]; then
    $WGET "${BASE_URL}Video-1080p&files=${r}.mp4" -O "${DIR_VIDEO_1080P}/${r}.mp4"
  fi
  if [ $DATA_VIDEO_576P -eq 1 ]; then
    $WGET "${BASE_URL}Video-576p&files=${r}.mp4" -O "${DIR_VIDEO_576P}/${r}.mp4"
  fi
done