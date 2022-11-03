#!/bin/bash

DIR_DATA="${PWD}/datasets/nina4"
DIR_RAW="${DIR_DATA}/raw"
DIR_PROC="${DIR_DATA}/processed"

python preprocess_nina4.py --path "${DIR_RAW}" --save "${DIR_PROC}" --ver 1 --mean --rectify --butter --ssize 50 --wsize 520 --rest
#python preprocess_nina4.py --path "${DIR_RAW}" --save "${DIR_PROC}" --ver 2 --mean --rectify --butter --ssize 50 --wsize 520 --rest
