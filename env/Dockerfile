FROM vllm/vllm-openai:latest

ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server --model $MODEL $VLLM_ARGS