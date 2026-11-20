"""Run or replay the A/B/C online bin-packing milestone experiment."""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
from statistics import mean
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from binpacking import BEST_FIT, FIRST_FIT, evaluate, weibull_instances


MODEL = "deepseek-flash"
ARMS = ("A", "B", "C")
CARD_FIELDS = ("rule", "change", "tradeoff")
OUTPUT_TOKENS = 420
SYSTEM = (
    "Design a deterministic online bin-packing priority function to minimize bin count. "
    "Your answer is one JSON object. Its expression field is a string containing one Python "
    "numeric expression of at most 500 characters. Variables are item, remaining and gap "
    "(remaining minus item), all normalized to [0, 1]. Only feasible bins are scored. Highest "
    "score wins; scores tie by opening order. Use + - * / %, comparisons, conditional "
    "expressions, and abs/min/max. Use no exponentiation, assignments or imports. Scores "
    "must be finite for all feasible pairs. Seek a new policy, not a rescaling of the parent."
)
ARM_PROMPTS = {
    "A": 'Use the JSON shape {"expression": "..."}.',
    "B": 'Use the JSON shape {"expression": "...", "strategy": "..."}. '
         "Write the strategy as free-form prose of at most 80 words.",
    "C": 'Use the JSON shape {"expression": "...", '
         '"strategy": {"rule": "...", "change": "...", "tradeoff": "..."}}. '
         "Describe the packing rule, change from the parent, and expected trade-off in "
         "those three fields. All fields combined may contain at most 80 words.",
}


@dataclass(frozen=True)
class Settings:
    """Experimental configuration shared across arms."""

    search_data_seed: int = 5491
    validation_data_seed: int = 5492
    search_seed: int = 2026
    search_instances: int = 10
    validation_instances: int = 20
    items: int = 1000
    token_budget: int = 20_000
    max_calls: int = 60

    def datasets(self):
        """Generate pilot search and validation datasets independently."""
        return (
            weibull_instances(self.search_data_seed, self.search_instances, self.items),
            weibull_instances(self.validation_data_seed, self.validation_instances, self.items),
        )


def complete(messages, api_key, max_tokens):
    """Make one DeepSeek request and return the content and measured usage."""
    body = {
        "model": MODEL, "messages": messages, "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"}, "max_tokens": max_tokens,
    }
    request = Request(
        "https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=75) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"DeepSeek API HTTP {exc.code} ({exc.reason}).") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"DeepSeek request failed: {exc}.") from exc
    usage = payload["usage"]
    choice = payload["choices"][0]
    return {
        "response_id": payload["id"], "model": payload["model"],
        "fingerprint": payload.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"),
        "content": choice["message"]["content"] or "",
        "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
        "cached_input_tokens": usage.get("prompt_cache_hit_tokens", 0),
    }


def parse_candidate(arm, content):
    """Validate the condition's response schema and strategy length."""
    data = json.loads(content)
    fields = {"expression"} if arm == "A" else {"expression", "strategy"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("Use exactly the requested JSON fields.")
    expression = data["expression"]
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("The expression must be a nonempty string.")
    strategy = data.get("strategy")
    if arm == "A":
        return expression.strip(), strategy
    if arm == "B" and isinstance(strategy, str) and strategy.strip():
        prose = strategy
    elif arm == "C" and isinstance(strategy, dict) and set(strategy) == set(CARD_FIELDS):
        values = [strategy[field] for field in CARD_FIELDS]
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("Every strategy field must be a nonempty string.")
        prose = " ".join(values)
    else:
        raise ValueError("Follow the requested strategy format.")
    if len(prose.split()) > 80 or len(prose) > 800:
        raise ValueError("Use at most 80 words and 800 characters of strategy text.")
    return expression.strip(), strategy


def make_messages(arm, parent, repair):
    """Build a mutation prompt with the parent and measured search feedback."""
    parent_data = {"expression": parent["expression"], "mean_search_bins": parent["score"]}
    if parent["strategy"] is not None:
        parent_data["strategy"] = parent["strategy"]
    prompt = f"{ARM_PROMPTS[arm]}\nParent: {json.dumps(parent_data, ensure_ascii=False)}"
    if repair is not None:
        prompt += f"\nRepair this invalid response ({repair['error']}): {repair['content'][:1600]}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def initial_candidate(search_data):
    """Make the common Best Fit starting program."""
    return {
        "id": 0, "expression": BEST_FIT, "strategy": None,
        "score": mean(evaluate(BEST_FIT, search_data)),
    }


def leaderboard(candidates):
    """Return the top five search candidates, breaking ties by discovery order."""
    return sorted(candidates.values(), key=lambda candidate: (candidate["score"], candidate["id"]))[:5]


def run_arm(arm, settings, search_data, destination, api_key):
    """Generate and score candidates, recording each request for key-free replay."""
    rng = random.Random(settings.search_seed)
    seed = initial_candidate(search_data)
    candidates = {BEST_FIT: seed}
    used = 0
    repair = None
    stop_reason = "call_limit"
    with destination.open("x", encoding="utf-8") as log:
        for index in range(1, settings.max_calls + 1):
            parent = repair["parent"] if repair else rng.choice(leaderboard(candidates))
            messages = make_messages(arm, parent, repair)
            # Reserve one token per UTF-8 byte plus room for API message framing.
            reserve = sum(len(message["content"].encode()) for message in messages) + 128
            if settings.token_budget - used < reserve + OUTPUT_TOKENS:
                stop_reason = "budget_reserve"
                break

            started = time.monotonic()
            event = complete(messages, api_key, OUTPUT_TOKENS)
            event.update(id=index, parent_id=parent["id"], repair=repair is not None)
            used += event["input_tokens"] + event["output_tokens"]
            try:
                expression, strategy = parse_candidate(arm, event["content"])
                score = mean(evaluate(expression, search_data))
                event.update(expression=expression, strategy=strategy, search_mean_bins=score)
                candidates.setdefault(expression, {
                    "id": index, "expression": expression, "strategy": strategy, "score": score,
                })
                repair = None
            except (ValueError, SyntaxError, ArithmeticError) as exc:
                event["error"] = str(exc)[:240]
                repair = None if repair else {**event, "parent": parent}
            event["elapsed_seconds"] = round(time.monotonic() - started, 3)
            print(json.dumps(event, ensure_ascii=False), file=log, flush=True)
            print(f"{arm} call {index}: {used} tokens, {event.get('error', 'valid')}", flush=True)
            if used > settings.token_budget:
                raise RuntimeError("Reported server usage exceeded the budget estimate.")
        print(json.dumps({"stop_reason": stop_reason}), file=log)


def evaluate_finalists(arm, events, search_data, validation_data):
    """Rebuild the shortlist and select the finalist using validation after search."""
    seed = initial_candidate(search_data)
    candidates = {BEST_FIT: seed}
    valid = [event for event in events if "expression" in event]
    for event in valid:
        expression, strategy = parse_candidate(arm, event["content"])
        score = mean(evaluate(expression, search_data))
        candidates.setdefault(expression, {
            "id": event["id"], "expression": expression, "strategy": strategy, "score": score,
        })
    finalists = [
        (evaluate(candidate["expression"], validation_data), candidate)
        for candidate in leaderboard(candidates)
    ]
    counts, winner = min(finalists, key=lambda pair: (mean(pair[0]), pair[1]["score"], pair[1]["id"]))
    return {
        "method": arm, "status": "evaluated" if valid else "no_valid_candidate",
        "mean_validation_bins": mean(counts), "validation_counts": counts,
        "winner": winner, "calls": len(events), "valid_candidates": len(valid),
        "input_tokens": sum(event["input_tokens"] for event in events),
        "output_tokens": sum(event["output_tokens"] for event in events),
        "fingerprints": sorted({event["fingerprint"] for event in events if event["fingerprint"]}),
        "models": sorted({event["model"] for event in events}),
        "elapsed_seconds": round(sum(event["elapsed_seconds"] for event in events), 3),
    }


def summarize(settings, run_directory=None):
    """Evaluate both baselines and replay any saved search attempts."""
    search_data, validation_data = settings.datasets()
    rows = []
    for method, expression in (("First Fit", FIRST_FIT), ("Best Fit", BEST_FIT)):
        counts = evaluate(expression, validation_data)
        rows.append({"method": method, "status": "evaluated", "mean_validation_bins": mean(counts),
                     "validation_counts": counts})
    best_fit = rows[1]["mean_validation_bins"]
    for arm in ARMS:
        log = run_directory / f"{arm}.jsonl" if run_directory else None
        if log is None or not log.exists():
            rows.append({"method": arm, "status": "pending"})
            continue
        lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        events = [event for event in lines if "id" in event]
        if not events:
            rows.append({"method": arm, "status": "no_completed_calls"})
            continue
        row = evaluate_finalists(arm, events, search_data, validation_data)
        row["stop_reason"] = lines[-1].get("stop_reason", "interrupted")
        if row["stop_reason"] == "interrupted":
            row["status"] = "interrupted"
        rows.append(row)
    for row in rows:
        if "mean_validation_bins" in row:
            row["relative_to_best_fit_pct"] = round((row["mean_validation_bins"] / best_fit - 1) * 100, 4)
    return {"settings": asdict(settings), "arms": rows}


def main():
    """Dispatch the baseline, live run or archived replay command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baselines", help="evaluate baselines without API access")
    baseline.add_argument("--output", type=Path)
    run = commands.add_parser("run", help="run all three conditions on the DeepSeek API")
    run.add_argument("--output", type=Path, required=True, help="new or initialized run directory")
    run.add_argument("--arm", choices=ARMS, help="run one condition only")
    run.add_argument("--token-budget", type=int, default=Settings.token_budget)
    run.add_argument("--max-calls", type=int, default=Settings.max_calls)
    replay = commands.add_parser("replay", help="recompute results from stored API responses")
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "baselines":
        settings = Settings()
        report = summarize(settings)
    elif args.command == "replay":
        settings = Settings(**json.loads((args.run / "settings.json").read_text(encoding="utf-8")))
        report = summarize(settings, args.run)
    else:
        if args.token_budget < 1 or args.max_calls < 1:
            parser.error("The token budget and call limit must be positive.")
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            parser.error("Set DEEPSEEK_API_KEY before running a live pilot.")
        settings = Settings(token_budget=args.token_budget, max_calls=args.max_calls)
        args.output.mkdir(parents=True, exist_ok=True)
        config_path = args.output / "settings.json"
        configuration = json.dumps(asdict(settings), indent=2) + "\n"
        try:
            with config_path.open("x", encoding="utf-8", newline="\n") as config:
                config.write(configuration)
        except FileExistsError:
            if json.loads(config_path.read_text(encoding="utf-8")) != asdict(settings):
                parser.error("The output directory already has different pilot settings.")
        search_data, _ = settings.datasets()
        selected_arms = (args.arm,) if args.arm else ARMS
        for arm in selected_arms:
            destination = args.output / f"{arm}.jsonl"
            if destination.exists() and not args.arm:
                lines = destination.read_text(encoding="utf-8").splitlines()
                if not lines or "stop_reason" not in json.loads(lines[-1]):
                    parser.error(f"{destination} is incomplete. Replay it or choose a new run directory.")
                print(f"{arm}: using existing log {destination}", flush=True)
                continue
            run_arm(arm, settings, search_data, destination, api_key)
        report = summarize(settings, args.output)
        args.output /= "summary.json"

    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8", newline="\n")
        print(f"Wrote {args.output}")
    else:
        print(output, end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
