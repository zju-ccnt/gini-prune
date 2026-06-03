total_path = {
        "llama2-7b": '~/.cache/modelscope/hub/models/LLM-Research/llama-2-7b',
        "qwen3-4b": '~/.cache/modelscope/hub/models/Qwen/Qwen3-4B',
        "qwen3-8b": '~/.cache/modelscope/hub/models/Qwen/Qwen3-8B',
        "mistral-7b": '~/.cache/modelscope/hub/models/AI-ModelScope/Mistral-7B-v0.1'
}
protect_tail = {
        "llama2-7b": 2,
        "qwen3-4b": 1,
        "qwen3-8b": 1,
        "mistral-7b": 2
}
protect_head = 5
pruner = {
        "llama2-7b": 9,
        "qwen3-4b": 8,
        "qwen3-8b": 9,
        "mistral-7b": 9
}
total_rank = {
    "llama2-7b": 32,
    "qwen3-4b": 4,
    "qwen3-8b": 2,
    "mistral-7b": 2
}

choose_model = "qwen3-8b"
only_pruner = False