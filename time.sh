python3 -m flexllmgen.flex_opt --model facebook/opt-6.7b \
    --percent 50 50 50 50 100 0 \
    --verbose 10 \
    --num-gpu-batches 8 \
    --gpu-batch-size 4 \
    --debug-mode breakdown