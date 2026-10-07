"""Run a public-PyTAG Pandemic pilot twice in independent Python processes.

Run from the PyTAG repository with PYTAG_JAR_PATH pointing at TAG-pytag.jar.
Each worker changes to the TAG checkout so data/pandemic resolves correctly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TAG_ROOT = ROOT / "pytag" / "TabletopGames"
FIELDS = ("run", "python_players", "seed", "outcome", "decision_count", "runtime_seconds", "action_trace", "exception")
DIFF_FIELDS = ("python_players", "seed", "field", "run_a", "run_b")


def _cases(games: int):
    single = games // 2
    return [(1, seed) for seed in range(single)] + [
        (2, seed) for seed in range(games - single)
    ]


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\r", " ").replace("\n", " ")


def _outcome(env) -> str:
    reward = env.terminal_rewards()[0]
    outcome = {1.0: "win", -1.0: "loss", 0.5: "draw"}.get(reward)
    if outcome is None:
        raise AssertionError(f"terminal state has no win/loss/draw player result (reward={reward})")
    return outcome


def _play(env, python_players: int, seed: int, max_decisions: int, progress: list[int]) -> tuple[str, int]:
    import numpy as np

    rng = random.Random(seed)
    _, info = env.reset(seed=seed)
    for decision in range(max_decisions):
        progress[0] = decision
        player = env.getPlayerID()
        current = info[player] if python_players == 2 else info
        actions = current["legal_actions"]
        mask = current["action_mask"]

        assert actions, f"no legal actions at decision {decision} for player {player}"
        ids = [action["id"] for action in actions]
        assert len(ids) == len(set(ids)), f"duplicate action ID at decision {decision}"
        indices = [action["mask_index"] for action in actions]
        assert len(indices) == len(set(indices)), f"duplicate mask position at decision {decision}"
        assert all(0 <= index < len(mask) and bool(mask[index]) for index in indices), (
            f"listed action has an illegal mask position at decision {decision}"
        )
        assert set(indices) == set(np.flatnonzero(mask).tolist()), (
            f"legal list and mask differ at decision {decision}"
        )

        selected = rng.choice(actions)
        trace_position = len(env.get_action_trace())
        try:
            _, _, done, info = env.step(selected["id"])
        except Exception as exc:
            raise RuntimeError(
                f"step failed at decision {decision} for player {player}, "
                f"action {selected['id']}: {_error(exc)}"
            ) from exc
        trace = env.get_action_trace()
        assert len(trace) > trace_position and trace[trace_position] == selected["id"], (
            f"selected ID did not execute at decision {decision}: {selected['id']}"
        )
        progress[0] = decision + 1
        if done:
            return _outcome(env), decision + 1
    raise AssertionError(f"game exceeded {max_decisions} Python decisions")


def _worker(run: str, output: Path, games: int, max_decisions: int) -> None:
    # Import through the installed/local public Python package only.
    sys.path.insert(0, str(ROOT))
    os.chdir(TAG_ROOT)
    from pytag import MultiAgentPyTAG, PyTAG

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for python_players in (1, 2):
            seeds = [seed for players, seed in _cases(games) if players == python_players]
            try:
                agents = ["python"] * python_players + ["random"] * (2 - python_players)
                cls = MultiAgentPyTAG if python_players == 2 else PyTAG
                env = cls(agents, game_id="Pandemic", obs_type="json", seed=0)
            except Exception as exc:
                for seed in seeds:
                    writer.writerow(dict(run=run, python_players=python_players, seed=seed,
                                         outcome="error", decision_count=0, runtime_seconds="0.000000", action_trace="[]",
                                         exception=_error(exc)))
                    handle.flush()
                continue

            for seed in seeds:
                start = time.perf_counter()
                outcome, decisions, error = "error", 0, ""
                progress = [0]
                try:
                    outcome, decisions = _play(env, python_players, seed, max_decisions, progress)
                except Exception as exc:
                    error = _error(exc)
                    decisions = progress[0]
                trace = json.dumps(env.get_action_trace(), ensure_ascii=False)
                writer.writerow(dict(run=run, python_players=python_players, seed=seed,
                                     outcome=outcome, decision_count=decisions,
                                     runtime_seconds=f"{time.perf_counter() - start:.6f}",
                                     action_trace=trace,
                                     exception=error))
                handle.flush()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _compare(a: list[dict[str, str]], b: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {(row["python_players"], row["seed"]): row for row in b}
    differences = []
    for row in a:
        key = (row["python_players"], row["seed"])
        other = by_key.get(key)
        for field in ("outcome", "decision_count", "action_trace", "exception"):
            left, right = row[field], other[field] if other is not None else "MISSING"
            if left != right:
                differences.append(dict(python_players=key[0], seed=key[1], field=field,
                                        run_a=left, run_b=right))
    return differences


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "pandemic_pilot.csv")
    parser.add_argument("--games", type=int, default=100, help="Games per process; default: 100")
    parser.add_argument("--max-decisions", type=int, default=2000)
    parser.add_argument("--worker", choices=("a", "b"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.games < 2 or args.max_decisions < 1:
        parser.error("--games must be at least 2 and --max-decisions must be positive")
    jar = os.environ.get("PYTAG_JAR_PATH")
    if not jar or not Path(jar).is_file():
        parser.error("set PYTAG_JAR_PATH to the built target/TAG-pytag.jar")
    if not (TAG_ROOT / "data" / "pandemic").is_dir():
        parser.error(f"TAG data/pandemic is missing under {TAG_ROOT}")
    output = args.output.resolve()
    if args.worker:
        _worker(args.worker, output, args.games, args.max_decisions)
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    run_files = {run: output.with_name(f"{output.stem}_run_{run}{output.suffix}") for run in ("a", "b")}
    elapsed = {}
    for run in ("a", "b"):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", run,
                   "--games", str(args.games), "--max-decisions", str(args.max_decisions),
                   "--output", str(run_files[run])]
        start = time.perf_counter()
        result = subprocess.run(command, cwd=TAG_ROOT, text=True, capture_output=True)
        elapsed[run] = time.perf_counter() - start
        if result.returncode:
            print(result.stderr or result.stdout, file=sys.stderr)
            print(f"Worker {run} failed; partial CSV: {run_files[run]}", file=sys.stderr)
            return result.returncode

    a, b = _read_rows(run_files["a"]), _read_rows(run_files["b"])
    if len(a) != args.games or len(b) != args.games:
        print(f"Expected {args.games} rows per run, got {len(a)} and {len(b)}", file=sys.stderr)
        return 1
    _write_csv(output, FIELDS, a + b)
    differences = _compare(a, b)
    difference_file = output.with_name(f"{output.stem}_differences{output.suffix}")
    _write_csv(difference_file, DIFF_FIELDS, differences)

    for run, rows in (("a", a), ("b", b)):
        failures = sum(bool(row["exception"]) for row in rows)
        completed = len(rows) - failures
        print(f"Run {run}: {completed}/{len(rows)} completed, {failures} failures, "
              f"{completed * 60 / elapsed[run]:.1f} completed games/minute "
              f"({elapsed[run]:.1f}s process wall time)")
    changed_pairs = len({(row["python_players"], row["seed"]) for row in differences})
    print(f"Reproducibility: {changed_pairs}/{args.games} seed/mode pairs differ "
          f"({len(differences)} field differences; runtimes excluded)")
    print(f"Records: {output}")
    print(f"Differences: {difference_file}")
    return 1 if any(row["exception"] for row in a + b) else 0


if __name__ == "__main__":
    raise SystemExit(main())
