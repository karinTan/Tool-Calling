#qwen3-4b-think-2507
python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/qwen3-4b-think-local \
    --served-model-name qwen3-4b-think-local \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768

bfcl generate --model qwen3-4b-think-local --test-category multi_turn --skip-server-setup
bfcl evaluate --model qwen3-4b-think-local --test-category multi_turn --partial-eval

#qwen3-4b-think-mine
python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/qwen3-4b-think-local \
    --served-model-name qwen3-4b-think-mine \
    --gpu-memory-utilization 0.95 \
    --max-model-len 98304 \
    --enable-lora \
    --lora-modules lora=/root/autodl-tmp/zhipu/saves/Qwen3-4B-Thinking-2507/lora/train_2026-01-27-01-18-41/checkpoint-693 \
    --tensor-parallel-size 2 \
    --max-num-seqs 2 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --chat-template /root/autodl-tmp/zhipu/qwen.jinja \
    --reasoning-parser qwen3

bfcl generate --model qwen3-4b-think-mine --test-category multi_turn_base --skip-server-setup
bfcl generate --model qwen3-4b-think-mine --test-category multi_turn_long_context --skip-server-setup
bfcl generate --model qwen3-4b-think-mine --test-category multi_turn_miss_func --skip-server-setup
bfcl generate --model qwen3-4b-think-mine --test-category multi_turn_miss_param --skip-server-setup

bfcl evaluate --model qwen3-4b-think-mine --test-category multi_turn --partial-eval