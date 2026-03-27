#qwen3-4b-mine
python -m vllm.entrypoints.openai.api_server \
    --model /root/qwen3-4b-think-local \
    --served-model-name qwen3-4b-think-mine \
    --gpu-memory-utilization 0.9 \
    --max-model-len 40960 \
    --enable-lora \
    --lora-modules lora=/root/autodl-tmp/zhipu/saves/Qwen3-4B-Thinking-2507/lora/train_2026-01-27-01-18-41/checkpoint-693
    
#qwen3-4b-think-2507
python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/qwen3-4b-think-local \
    --served-model-name qwen3-4b-think-local \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768

bfcl generate --model qwen3-4b-think-local --test-category multi_turn --skip-server-setup
bfcl evaluate --model qwen3-4b-think-local --test-category multi_turn --partial-eval