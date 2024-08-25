#!/bin/bash
set -e


if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Must call with ./run.sh <metric_path> "
    exit 1
fi

metric_path=$1

echo ${test_name} is writing the benchmark result to ${metric_path}

pip install diffusers==0.20.2 transformers==4.33.1 accelerate==0.22.0 safetensors==0.3.1 matplotlib
python3 sd2_512_compile.py
python3 sd2_512_benchmark.py $1