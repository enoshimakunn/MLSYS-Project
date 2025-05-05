python3 -m flexllmgen.flex_opt --model facebook/opt-1.3b \
    --percent 20 80 20 80 100 0 \
    --verbose 10 \
    --num-gpu-batches 1 \
    --gpu-batch-size 1 \
    --cpu-cache-compute