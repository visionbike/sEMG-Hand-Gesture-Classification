#!/bin/bash

DB_NAME="nina1"
DIR_BASE="${PWD}/datasets/${DB_NAME}/raw"
BASE_URL="https://www.dropbox.com/s/xj5mx346t9j00lu/"

if [ ! -d "$DIR_BASE" ]; then
  mkdir -p "${DIR_BASE}"
fi

WGET="wget -c -N"

echo "### Downloading NinaPro DB1..."
$WGET "${BASE_URL}${DB_NAME}.zip?dl=1" -O "${DIR_BASE}/${DB_NAME}.zip"
unzip -q "${DIR_BASE}/${DB_NAME}.zip" -d "${DIR_BASE}/${DB_NAME}"
rm -f "${DIR_BASE}/${DB_NAME}.zip"

for i in {1..27}; do
  unzip -q "${DIR_BASE}/${DB_NAME}/s${i}.zip" -d "${DIR_BASE}/s${i}"
done

# shellcheck disable=SC2115
rm -rf "${DIR_BASE}/${DB_NAME}"
