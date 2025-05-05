python3 -m flexllmgen.flex_opt --model facebook/opt-6.7b \
    --percent 20 80 20 80 100 0 \
    --verbose 10 \
    --num-gpu-batches 8 \
    --gpu-batch-size 1 \
    --debug-mode breakdown


python3 -m flexllmgen.flex_opt_prefetch --model facebook/opt-6.7b \
    --percent 20 80 20 80 100 0 \
    --verbose 10 \
    --num-gpu-batches 8 \
    --gpu-batch-size 1 \
    --debug-mode breakdown