# Thoughts Augmented FunSearch for Online Bin Packing

Course project milestone. The experiment asks whether carrying a structured strategy card alongside a candidate program improves the heuristics found under the same LLM token budget. A uses code only, B includes a free-form description, and C includes three fields for the rule, the change from the parent, and its expected trade-off.

## Environment

Use Python 3.12. The runner needs only the standard library and runs on a CPU. Run the commands below from the directory containing `pilot.py`.

## Reproduce the measured baselines

```text
python pilot.py baselines --output runs/baselines.json
```

This takes a few seconds. On 20 pilot validation sequences of 1,000 items, the mean bin counts are **424.20 for First Fit** and **422.35 for Best Fit**. The tracked result is [`results/baselines.json`](results/baselines.json). The generator draws from Weibull with scale 45 and shape 3, rounds to integers in 1–100, and uses bin capacity 100, following the distribution in the [FunSearch supplement, Appendix E.4](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf). Seeds 5491 and 5492 generate 10 search and 20 validation sequences, respectively.

## Run A/B/C with DeepSeek

Set the `DEEPSEEK_API_KEY` environment variable using an [API key from DeepSeek](https://platform.deepseek.com/api_keys).

```text
python pilot.py run --output runs/pilot --token-budget 20000 --max-calls 60
python pilot.py replay --run runs/pilot --output runs/replayed.json
```

The first command uses `deepseek-flash` with thinking disabled. Each arm has the same 20,000-token allowance and 60-call ceiling. The runner logs progress per call. Runtime depends on API latency and token use. With a multi-second response time, allow several minutes. For a quick integration check, use a different output directory and `--max-calls 2`.

The runner uses Best Fit as the initial program and a seeded random choice from the five best search candidates as the parent. The same constrained Python expression interface is used across arms. The evaluator handles feasibility and breaks score ties by bin opening order. Only mean search bin count determines the leaderboard. One repair request is allowed after an invalid response, with the same accounting as a new proposal. After search, the validation set selects a finalist from the five candidates on the leaderboard. The formal test set is reserved for the final experiments.

Each arm writes a JSONL log with the candidate response, parent identifier, validation of the response format, search score or error, API model and fingerprint, elapsed time, and counted input and output tokens. Each run records its settings and a `summary.json` with validation counts and results relative to Best Fit. Replay needs no API key. It rebuilds the search scores and recalculates finalist scores from the archived model responses. Fresh API calls can return different candidates. For a submitted live result, archive the run's `settings.json` and A/B/C JSONL logs under `results/` and use that directory as `--run` to reproduce the report numbers.

`--arm A`, `--arm B`, or `--arm C` runs an individual condition within the same experiment settings. Completed arm logs are reused when starting the three-arm command again. If an API failure interrupts an arm, its log remains available for replay, and a fresh run should use a new output directory.
