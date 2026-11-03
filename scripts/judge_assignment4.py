#!/usr/bin/env python3
"""Score every Assignment 4 failure boundary with a local Ollama judge."""

import argparse
import json
from pathlib import Path

from evaloop.analysis.similarity import (
    DEFAULT_JUDGE_PROMPT,
    SemanticSimilarityJudge,
    collect_similarity_scores,
    default_similarity_path,
    extract_boundary_pair,
)
from evaloop.models.ollama import OllamaLLM


def main() -> None:
    """Judge and cache the semantic similarity of all recorded boundaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--model", default="qwen3.8")
    parser.add_argument("--judge-name", default="qwen35-27b-local")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    paths = sorted(args.results_dir.resolve().glob("*_results.json"))
    if not paths:
        raise SystemExit(f"no result JSON files found under {args.results_dir}")

    llm = OllamaLLM(
        {
            "name": args.judge_name,
            "model_id": args.model,
            "base_url": args.base_url,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 16,
            "seed": 42,
            "timeout": 600,
            "keep_alive": "30m",
        }
    )
    judge = SemanticSimilarityJudge(
        llm,
        name=args.judge_name,
        prompt_template=DEFAULT_JUDGE_PROMPT,
    )

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or len(payload) != 1:
            raise ValueError(f"Expected one model result in {path}")
        result = payload[0]
        max_cycles = int(result.get("max_cycles", 10))
        pairs = [
            pair
            for task in result["prompt_results"]
            if (pair := extract_boundary_pair(task, max_cycles)) is not None
        ]
        cache_path = default_similarity_path(path, args.judge_name)
        scores = collect_similarity_scores(
            pairs,
            cache_path,
            lambda: judge,
            prompt_template=DEFAULT_JUDGE_PROMPT,
        )
        usable = [score for score in scores.values() if score is not None]
        mean = sum(usable) / len(usable) if usable else float("nan")
        print(
            f"{path.name}: {len(usable)}/{len(pairs)} scored, "
            f"mean={mean:.4f}, cache={cache_path.name}"
        )


if __name__ == "__main__":
    main()
