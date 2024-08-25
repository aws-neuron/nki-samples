#!/bin/bash
set -e


if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Must call with ./run.sh <metric_path> "
    exit 1
fi

metric_path=$1

echo ${test_name} is writing the benchmark result to ${metric_path}

export NEURON_FUSE_SOFTMAX=1
pip install diffusers==0.24.0 transformers==4.35.2 accelerate==0.24.1 safetensors==0.4.1 matplotlib
python3 sd2_inpainting_936_624_compile.py
python3 sd2_inpainting_936_624_benchmark.py $1