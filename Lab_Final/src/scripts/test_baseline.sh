export CUDA_VISIBLE_DEVICES=0

cd ../

python3 main.py \
    --exp_name baseline \
    # --model qwen-long