#!/bin/bash

DIR_DATA="${PWD}/datasets/nina4"
DIR_RAW="${DIR_DATA}/raw"
DIR_PROC="${DIR_DATA}/processed"

python preprocess_nina4.py --path "${DIR_RAW}" --save "${DIR_PROC}" --rectify --butter --ssize 50 --wsize 520 --first --rest
#python preprocess_nina4.py --path "${DIR_RAW}" --save "${DIR_PROC}" --rectify --butter --ulaw --ssize 50 --wsize 520 --first --rest
#python preprocess_nina4.py --path "${DIR_RAW}" --save "${DIR_PROC}" --rectify --butter --minmax --ssize 50 --wsize 520 --first --rest
#python preprocess_nina4.py --path "${DIR_RAW}" --save "${DIR_PROC}" --rectify --butter --ulaw --minmax --ssize 50 --wsize 520 --first --rest
