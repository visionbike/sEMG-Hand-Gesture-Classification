#!/bin/bash

DIR_DATA="${PWD}/datasets/nina1"
DIR_RAW="${DIR_DATA}/raw"
DIR_PROC="${DIR_DATA}/processed"

#python preprocess_nina1.py --path "${DIR_RAW}" --save "${DIR_PROC}" --ver 1 --mean --rectify --butter --ssize 2 --wsize 26 --rest
python preprocess_nina1.py --path "${DIR_RAW}" --save "${DIR_PROC}" --ver 2 --mean --rectify --butter --ssize 2 --wsize 26 --rest

