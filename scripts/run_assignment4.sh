#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
OUTPUT_DIR="${PROJECT_DIR}/results/assignment4"
MODEL_7B="${MODEL_7B:-qwen2.5-coder:7b-instruct-q4_K_M}"
MODEL_27B="${MODEL_27B:-qwen3.8}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}"

run_one() {
    local model="$1"
    local loop_type="$2"
    local experiment_name="$3"
    local result_file="${OUTPUT_DIR}/${experiment_name}_code_generation_summarization_results.json"

    if [[ -s "${result_file}" ]]; then
        echo "Skipping completed run: ${result_file}"
        return
    fi

    "${PYTHON_BIN}" -m evaloop evaluate \
        --model="${model}" \
        --loop_type="${loop_type}" \
        --dataset=mbpp_plus \
        --task_limit=100 \
        --task_seed=42 \
        --max_cycles=10 \
        --temperature=0.0 \
        --top_p=1.0 \
        --max_new_tokens=1024 \
        --seed=42 \
        --output_dir="${OUTPUT_DIR}" \
        --experiment_name="${experiment_name}"
}

run_one "ollama/${MODEL_7B}" standard qwen25_coder_7b_standard
run_one "ollama/${MODEL_7B}" chinese qwen25_coder_7b_chinese
run_one "ollama/${MODEL_27B}" standard qwen35_27b_standard
run_one "ollama/${MODEL_27B}" chinese qwen35_27b_chinese

"${PYTHON_BIN}" scripts/judge_assignment4.py "${OUTPUT_DIR}" \
    --model "${MODEL_27B}" \
    --judge-name qwen35-27b-local
"${PYTHON_BIN}" scripts/summarize_assignment4.py "${OUTPUT_DIR}" \
    --judge-name qwen35-27b-local
