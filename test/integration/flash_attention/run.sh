#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Must call with ./run.sh <metric_path> "
    exit 1
fi

metric_path=$1

echo ${test_name} is writing the benchmark result to ${metric_path}

pytest flash_attention_correctness.py
python flash_attention_benchmark.py $metric_path

