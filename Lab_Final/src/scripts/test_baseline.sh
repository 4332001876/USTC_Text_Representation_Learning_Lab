export CUDA_VISIBLE_DEVICES=0

cd ../

nohup python3 main.py \
    --exp_name baseline \
    > baseline.log &
    # --model qwen-long