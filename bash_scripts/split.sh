#!/usr/bin/env bash

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for f in $1/*; do
	python "${dir}"../test/train_test_split.py "${f}"
done
