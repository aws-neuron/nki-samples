#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:$PWD

echo Creating benchmark result json at $PWD/test_result.json
if [ -f $PWD/test_result.json ]; then
    rm $PWD/test_result.json
fi
touch $PWD/test_result.json
echo "[]" >> $PWD/test_result.json

RESULT_JSON=$PWD/test_result.json

pushd fused_sd_attention_small_head
sh run.sh ${RESULT_JSON}
popd

pushd resize_nearest_fixed_dma_kernel
sh run.sh ${RESULT_JSON}
popd

pushd flash_attention
sh run.sh ${RESULT_JSON}
popd

pushd select_and_scatter_kernel 
sh run.sh ${RESULT_JSON}
popd
