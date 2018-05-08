#!/usr/bin/env bash

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for f in $1/*; do
	python "${dir}"/../py_scripts/train_test_split.py "${f}"
done

python "${dir}"/../py_scripts/tt_move.py "${1}"
