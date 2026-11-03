# Cross-Lingual Self-Description as an EvaLoop Robustness Attack

*CSC 591/791 Trustworthy AI for Software Engineering — Assignment 4*

## Extension and research question

I added an Option B loop in which a model alternates between Python code and a one-sentence Chinese description of that code. The research question is: **Does a Chinese self-description boundary change code-generation robustness relative to the standard English boundary, and does that effect depend on model size?** I expected cross-lingual compression to lose requirements and reduce ASL, with a larger drop for the 7B model.

The code-generation prompt and test oracle are unchanged. After a passing generation, the new inverse-task prompt asks: `用一句中文总结以下 Python 代码实现的任务，并以“编写一个 Python 函数来”开头` and requires only that one sentence. Postprocessing wraps it in a docstring and appends the original example `assert` verbatim. Thus the next prompt contains only model-produced semantic text plus the same signature anchor as standard EvaLoop. A task fails when either its EvalPlus MBPP base or plus tests fail; otherwise it continues for at most ten cycles. A preliminary 7B pilot requested detailed inputs, outputs, and edge cases. I excluded it because it confounded language with summary detail (ASL_pow 5.772), then fixed the reported protocol to mirror the standard one-sentence prompt before running the controlled comparison.

## Framework pipeline and changes

For each MBPP Plus record, the evaluator gives the natural-language task and example assert to `CodeGenerationTask`, which builds the model prompt and extracts raw or fenced Python from the answer. EvalPlus `untrusted_check` executes that code in its restricted worker against both the original and enhanced tests. Failure ends the task. After a pass, `CodeSummarizationTask` presents the generated code to the same model. Its postprocessor converts the returned sentence into the next docstring and restores the original assert. This retained line prevents trivial drift in function name and call signature, although one example cannot encode every hidden requirement. The evaluator records every prompt, code, summary, test outcome, and `successful_cycles` value.

For bucket `i` (tasks sustaining exactly `i` loops), EvaLoop's headline score is `ASL = sum(n_i i^2 s_i)/(T M)`, where `T=100`, `M=10`, and `s_i` averages similarity 1 for successful transitions with the judged similarity of the final failure boundary. I added the Chinese task in a new file and registry entry; CLI/config options for loop type and seeded task sampling; an Ollama backend; result metadata and atomic checkpoints; and reproduction, scoring, and failure-export scripts. Existing code-generation and test components were not changed. Unit tests cover prompt construction, assert retention, model-factory behavior, and Ollama request/response handling.

## Experimental setup

I evaluated local quantized **Qwen2.5-Coder-7B-Instruct Q4_K_M** and **Qwen3.5-27B Q4_K_M** (local Ollama alias `qwen3.8`) on the same 100 tasks sampled by `random.Random(42).sample` from EvalPlus's MBPP Plus order. Each model ran standard and Chinese loops once with temperature 0, top-p 1, seed 42, at most 1,024 output tokens and ten cycles. Inference ran through Ollama on a server with two NVIDIA RTX 6000 Ada 48 GB GPUs; the models fit locally and no paid API was used.

Headline ASL uses the framework's unchanged `DEFAULT_JUDGE_PROMPT`: it shows the two prompts and generated codes around a failure, asks for semantic similarity, defines 1.0 as equivalent and 0.0 as completely different, and requests only a score in `[0,1]`. The local 27B model judged all 23 post-first-cycle boundaries with temperature 0, top-p 1, seed 42, and 16 output tokens. The first parsed number is clamped to `[0,1]`, cached, grouped by failure bucket, and converted to `s_i` by the framework. Total and per-example API cost were **$0**. Because this judge differs from the paper's GPT-4 Turbo judge, these ASLs support only within-study comparisons. The 23 qualitative labels were assigned by direct record/test inspection, not by an LLM classifier.

## Results and failure analysis

```text
Model  Loop      Success   ASL      ASL_pow  Loop 10
7B     Standard  77%       6.7171   6.7250   67%
7B     Chinese   77%       7.2043   7.2080   72%
27B    Standard  79%       7.7043   7.7050   77%
27B    Chinese   79%       7.3066   7.3090   73%
```

Success rates are identical because cycle 1 precedes either summarization attack. The 27B model remains first, but its lead over 7B shrinks from 0.9873 ASL in English to 0.1023 in Chinese—an 89.6% reduction. Thus the standard loop's apparent size advantage is not language-invariant. The extension helps 7B yet hurts 27B, contradicting my directional hypothesis and revealing an interaction that either aggregate pass@1 or English-only looping would miss.

I manually inspected every task that passed cycle 1 but failed before cycle 10: 23 boundaries (10 standard + 5 Chinese for 7B; 2 standard + 6 Chinese for 27B). Seventeen (73.9%) were specification drift: output-role changes, inversions, added concepts, or omitted/narrowed constraints. Six (26.1%) preserved an adequate summary but regenerated a wrong formula, algorithm, comparator, or boundary check. There were no formatting/extraction failures. The retained assert often under-specified the distinction: for `Mbpp/238`, 27B Chinese added “distinct” to “all substrings”; a set implementation passes `"abc" == 6` but fails enhanced repeated-character cases. Other representative drifts changed common *elements* into their *indices* (`Mbpp/754`), subsequence into substring (`Mbpp/247`), and non-digits into letters (`Mbpp/619`, dropping punctuation).

Failure modes differed by model and boundary rather than by a simple language-wide count: standard and Chinese produced 12 and 11 later failures overall, but Chinese cut 7B's count from 10 to 5 and raised 27B's from 2 to 6. A case the extension **did not break** is `Mbpp/555`: 7B English reversed `sum(cubes)-sum(values)` after four successful cycles, while its Chinese wording kept the operand order and survived all ten. A case the extension **broke** is `Mbpp/238`: 27B English retained the closed-form count for ten cycles, while Chinese introduced “distinct” at the first summary and failed cycle 2. Shared 7B failures (`Mbpp/754`, `/773`, `/790`) show that the assert anchor cannot repair a semantic distinction absent from its single example.

## Limitations and implications

The evidence covers two related Qwen models, one 100-task seeded subset, one deterministic run, English source tasks, and quantized Ollama builds; it cannot establish a general Chinese or model-size effect. MBPP contamination is plausible, and even EvalPlus may miss wrong programs. The same 27B model served as evaluator and judge, which may favor its own style, especially cross-lingually; absolute ASL is not comparable with the public GPT-4-Turbo-scored leaderboard. The prompt was redesigned after a disclosed pilot, so future work should preregister it and replicate with other languages, families, seeds, and an independent multilingual judge. Still, the opposing ASL changes and concrete requirement mutations show why trustworthy-code evaluation should vary the representation boundary: robustness rankings can reflect the model-language interface, not only coding ability.
