#!/bin/bash

DIR_DATA="${PWD}/datasets/nina1"
DIR_RAW="${DIR_DATA}/raw"
DIR_PROC="${DIR_DATA}/processed"

python preprocess_nina1.py --path "${DIR_RAW}" --save "${DIR_PROC}" --rectify --butter --ssize 2 --wsize 26 --first --rest
#python preprocess_nina1.py --path "${DIR_RAW}" --save "${DIR_PROC}" --rectify --butter --ulaw --ssize 2 --wsize 26 --first --rest
#python preprocess_nina1.py --path "${DIR_RAW}" --save "${DIR_PROC}" --rectify --butter --minmax --ssize 2 --wsize 26 --first --rest
#python preprocess_nina1.py --path "${DIR_RAW}" --save "${DIR_PROC}" --rectify --butter --ulaw --minmax --ssize 2 --wsize 26 --first --rest