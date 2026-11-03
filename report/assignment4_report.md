# Chinese Summaries in the EvaLoop Loop

*CSC 591/791 Trustworthy AI for Software Engineering, Assignment 4*

## Extension and research question

I chose Option B. In the standard loop, the model writes code and then summarizes that code in one English sentence; the summary becomes the next task. In my loop, the model writes the summary in Chinese instead. My research question: **does a Chinese summary step change ASL compared with the standard English step, and is the effect different for a 7B and a 27B model?** I expected Chinese summaries to lose more requirements and lower ASL, with a larger drop for the 7B model.

Code generation and testing are unchanged. Only the summary prompt changes. After code passes the tests, the model gets this prompt: `用一句中文总结以下 Python 代码实现的任务，并以“编写一个 Python 函数来”开头` ("Summarize the task this Python code solves in one Chinese sentence, starting with 'Write a Python function to'"), and it must return only that sentence. The framework puts the sentence in a docstring and appends the original example `assert` line, as the standard loop does. This keeps the function name and signature the same in every cycle. The model's own summary is still the only thing that changes between cycles. A task fails when its code fails either the MBPP base tests or the EvalPlus plus tests, and the loop stops after ten cycles.

Before the final runs, I tried a pilot with the 7B model that asked for a detailed Chinese summary (inputs, outputs, edge cases). Its ASL_pow was 5.772. I dropped it because it changed two things at once (language and level of detail), and I switched to a one-sentence prompt that matches the standard one.

## Framework pipeline and changes

For each MBPP Plus task, the evaluator passes the task text and its example assert to `CodeGenerationTask`. That task builds the prompt and pulls the Python code out of the model's answer (plain or inside a code fence). EvalPlus `untrusted_check` runs the code in a separate restricted process against the base and plus tests. If the code fails, the task ends. If it passes, `CodeSummarizationTask` sends the code back to the same model for a summary. Its postprocessing turns the summary into the next docstring and adds the original assert line again. That line stops the function name and arguments from drifting between cycles, so a failure means the meaning drifted rather than the name. One example assert cannot cover every hidden requirement, though. The evaluator saves every prompt, code, summary, test result, and the `successful_cycles` count per task.

Let `n_i` be the number of tasks that complete exactly `i` cycles. The framework computes `ASL = sum(n_i * i^2 * s_i) / (T * M)` with `T = 100` tasks and `M = 10` cycles. `s_i` is 1 for tasks that never fail; for tasks that fail, it is the judge's similarity score between the last passing summary and the original task. ASL_pow is the same formula with every `s_i = 1`.

My changes: a new Chinese summary task in its own file, registered with the task factory; CLI and config options for the loop type and seeded task sampling; an Ollama model backend; extra result metadata and per-cycle checkpoints; and scripts to run the experiments, score failures, and write `summary.csv`. I did not change code generation, code extraction, or testing. New unit tests cover the Chinese prompt, keeping the assert line, the model factory, and the Ollama requests.

## Experimental setup

Models: **Qwen2.5-Coder-7B-Instruct Q4_K_M** and **Qwen3.5-27B Q4_K_M** (local Ollama tag `qwen3.8`). Tasks: the same 100 MBPP Plus tasks for all runs, chosen with `random.Random(42).sample` over EvalPlus's task order. Each model ran the standard loop and the Chinese loop once, with temperature 0, top-p 1, seed 42, at most 1,024 output tokens, and at most ten cycles. Everything ran locally through Ollama on two NVIDIA RTX 6000 Ada 48 GB GPUs, so there was no API cost.

LLM judge: ASL needs a similarity score for each failed task. I used the framework's unchanged `DEFAULT_JUDGE_PROMPT`. It shows the prompts and code before and after the failure, and asks for one similarity score in `[0,1]` (1.0 = same meaning, 0.0 = completely different). The judge was the local 27B model (temperature 0, top-p 1, seed 42, 16 output tokens). It scored all 23 tasks that passed cycle 1 and failed later. The framework takes the first number in the answer, clamps it to `[0,1]`, caches it, and uses it for `s_i`. API cost was $0 in total and per example. Because the paper used GPT-4 Turbo as the judge, my ASL values can be compared with each other but not with the public leaderboard. I labeled the 23 failures by hand; no LLM was used for that.

## Results and failure analysis

```text
Model  Loop      Success   ASL      ASL_pow  Reach cycle 10
7B     Standard  77%       6.7171   6.7250   67%
7B     Chinese   77%       7.2043   7.2080   72%
27B    Standard  79%       7.7043   7.7050   77%
27B    Chinese   79%       7.3066   7.3090   73%
```

Success rates are the same for both loops because cycle 1 runs before any summary is written. The Chinese loop raised the 7B model's ASL by 0.49 and lowered the 27B model's ASL by 0.40. The 27B model stays ahead, but its lead drops from 0.99 to 0.10. This is the opposite of what I expected for the 7B model. The differences are small, though: the 7B model reaches cycle 10 on 5 more tasks in Chinese, and the 27B model on 4 fewer. With 100 tasks and one run per setting, I cannot rule out that these differences are noise.

I inspected every task that passed cycle 1 but failed before cycle 10: 23 tasks in total (7B: 10 standard, 5 Chinese; 27B: 2 standard, 6 Chinese). In 17 of them (73.9%), the summary changed the task: it changed what the function returns, reversed a condition, added a requirement, or dropped or narrowed one. In the other 6 (26.1%), the summary was fine but the new code had a wrong formula, algorithm, comparison, or boundary check. None failed because of formatting or code extraction. The example assert often did not catch the change. For `Mbpp/238`, the 27B Chinese summary changed "all substrings" to "distinct substrings"; code using a set still passes the example `"abc" == 6` but fails plus tests with repeated characters. Other examples: common *elements* became their *indices* (`Mbpp/754`), subsequence became substring (`Mbpp/247`), and non-digits became letters, which drops punctuation (`Mbpp/619`).

The two loops had about the same number of later failures overall (12 standard, 11 Chinese), but the split by model differs. Chinese cut the 7B count from 10 to 5 and raised the 27B count from 2 to 6. A case the Chinese loop **did not break** is `Mbpp/555`: the 7B English summary swapped `sum(cubes)-sum(values)` after four passing cycles, while the Chinese summary kept the order and passed all ten cycles. A case it **broke** is `Mbpp/238`: the 27B English loop kept the correct formula for ten cycles, while the first Chinese summary added "distinct" and the task failed at cycle 2. Three 7B tasks failed in both loops (`Mbpp/754`, `/773`, `/790`); in each, the example assert could not tell the right meaning from the changed one.

## Limitations and implications

The results come from two Qwen models, one 100-task sample, one run each, English source tasks, and quantized Ollama builds. They do not show a general effect of Chinese or of model size. MBPP may be in the models' training data, and EvalPlus tests can still miss wrong programs. The 27B model was both a tested model and the judge, which may bias its scores, especially across languages. My ASL values are not comparable with the public leaderboard, which used GPT-4 Turbo as the judge. I changed the prompt after the pilot, so a follow-up should fix the prompt before running, and repeat the study with other languages, model families, seeds, and a separate multilingual judge. Even so, the two models moved in opposite directions, and the inspected summaries show concrete requirement changes. Changing only the summary language can change robustness scores, so an English-only loop may not give the full picture.
