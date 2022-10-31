#!/bin/bash

DIR_BASE="${PWD}/datasets/nina5/raw"
BASE_URL="https://zenodo.org/record/1000116/files/"

WGET="wget -c -N"

if [ ! -d "$DIR_BASE" ]; then
  mkdir -p "${DIR_BASE}"
fi

echo "### Downloading NinaPro DB5..."

if [ ! -x /usr/bin/wget ] ;
then
  echo "ERROR: no wget." >&2
  exit 1
fi

if [ ! -x /usr/bin/unzip ] ;
then
  echo "ERROR: no unzip." >&2
  exist 1
fi

for i in {1..10}; do
  $WGET "${BASE_URL}s${i}.zip?download=1" -O "${DIR_BASE}/s${i}.zip"
  unzip -q "${DIR_BASE}/s${i}.zip" -d "${DIR_BASE}/s${i}"
  rm -f "${DIR_BASE}/s${i}.zip"
done