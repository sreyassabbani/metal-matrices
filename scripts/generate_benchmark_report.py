#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchStats:
    name: str
    n: int
    mean_ns: float
    ci_low_ns: float
    ci_high_ns: float
    std_dev_ns: float
    samples_ns: list[float]


def load_bench_stats(group_dir: Path) -> list[BenchStats]:
    stats: list[BenchStats] = []
    for entry in sorted(group_dir.iterdir()):
        if not entry.is_dir() or entry.name == "report":
            continue
        sample_path = entry / "new" / "sample.json"
        estimates_path = entry / "new" / "estimates.json"
        if not sample_path.exists() or not estimates_path.exists():
            continue

        with sample_path.open("r", encoding="utf-8") as f:
            sample = json.load(f)
        with estimates_path.open("r", encoding="utf-8") as f:
            estimates = json.load(f)

        iters = sample["iters"]
        times = sample["times"]
        samples_ns = [t / i for t, i in zip(times, iters)]
        if not samples_ns:
            continue

        mean = estimates["mean"]["point_estimate"]
        ci_low = estimates["mean"]["confidence_interval"]["lower_bound"]
        ci_high = estimates["mean"]["confidence_interval"]["upper_bound"]
        std_dev = statistics.stdev(samples_ns) if len(samples_ns) > 1 else 0.0

        stats.append(
            BenchStats(
                name=entry.name,
                n=len(samples_ns),
                mean_ns=mean,
                ci_low_ns=ci_low,
                ci_high_ns=ci_high,
                std_dev_ns=std_dev,
                samples_ns=samples_ns,
            )
        )

    return stats


def permutation_p_value(a: list[float], b: list[float], reps: int, seed: int) -> float:
    rng = random.Random(seed)
    obs = abs(statistics.fmean(a) - statistics.fmean(b))
    combined = a + b
    a_len = len(a)
    hits = 0
    for _ in range(reps):
        rng.shuffle(combined)
        a_perm = combined[:a_len]
        b_perm = combined[a_len:]
        diff = abs(statistics.fmean(a_perm) - statistics.fmean(b_perm))
        if diff >= obs:
            hits += 1
    return (hits + 1) / (reps + 1)


def cliffs_delta(a: list[float], b: list[float]) -> float:
    gt = 0
    lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    total = len(a) * len(b)
    if total == 0:
        return 0.0
    return (gt - lt) / total


def cliffs_label(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def format_ns(value: float) -> str:
    if value >= 1_000_000.0:
        return f"{value / 1_000_000.0:.3f} ms"
    if value >= 1_000.0:
        return f"{value / 1_000.0:.3f} µs"
    return f"{value:.3f} ns"


def render_markdown(
    stats: list[BenchStats],
    baseline_name: str,
    output: Path,
    group: str,
    alpha: float,
    permutation_reps: int,
) -> None:
    by_name = {s.name: s for s in stats}
    if baseline_name not in by_name:
        baseline_name = stats[0].name
    baseline = by_name[baseline_name]

    rows = []
    for s in sorted(stats, key=lambda x: (x.name != baseline_name, x.mean_ns)):
        speed = baseline.mean_ns / s.mean_ns
        ci_half_width_ns = (s.ci_high_ns - s.ci_low_ns) / 2.0
        if s.name == baseline_name:
            p_value = "-"
            significant = "-"
            effect = "-"
        else:
            p = permutation_p_value(
                baseline.samples_ns,
                s.samples_ns,
                reps=permutation_reps,
                seed=42,
            )
            delta = cliffs_delta(baseline.samples_ns, s.samples_ns)
            p_value = f"{p:.4f}"
            significant = "yes" if p < alpha else "no"
            effect = f"{delta:+.3f} ({cliffs_label(delta)})"

        rows.append(
            "| "
            + " | ".join(
                [
                    s.name,
                    str(s.n),
                    f"{format_ns(s.mean_ns)} ± {format_ns(ci_half_width_ns)}",
                    format_ns(s.std_dev_ns),
                    f"{speed:.3f}x",
                    p_value,
                    significant,
                    effect,
                ]
            )
            + " |"
        )

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    md = "\n".join(
        [
            "# Benchmark Comparison",
            "",
            f"- Generated: {timestamp}",
            f"- Criterion group: `{group}`",
            f"- Baseline: `{baseline_name}`",
            f"- Significance: permutation test on per-sample ns/op (`alpha={alpha}`, `reps={permutation_reps}`)",
            "",
            "| Benchmark | n | μ ± 95% CI | σ | Speed vs baseline | p-value vs baseline | Significant | Cliff's delta |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            *rows,
            "",
            "_Lower mean time is faster._",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a condensed Markdown benchmark comparison table from Criterion output."
    )
    parser.add_argument(
        "--group",
        default="matrix multiplication",
        help="Criterion benchmark group directory under target/criterion.",
    )
    parser.add_argument(
        "--baseline",
        default="cpu matrix multiply",
        help="Benchmark name to use as baseline for relative speed and p-values.",
    )
    parser.add_argument(
        "--criterion-root",
        default="target/criterion",
        help="Criterion output directory root.",
    )
    parser.add_argument(
        "--output",
        default="docs/benchmark-comparison.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for the permutation test.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=20000,
        help="Number of permutation test resamples.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run cargo bench; only parse existing Criterion output.",
    )
    args = parser.parse_args()

    if not args.no_run:
        criterion_root = Path(args.criterion_root)
        group_dir = criterion_root / args.group
        if group_dir.exists():
            shutil.rmtree(group_dir)
        subprocess.run(["just", "all"], check=False)
        subprocess.run(
            ["cargo", "bench", "--bench", "matrix_multiplication"],
            check=True,
        )

    criterion_root = Path(args.criterion_root)
    group_dir = criterion_root / args.group
    if not group_dir.exists():
        raise SystemExit(f"Criterion group directory not found: {group_dir}")

    stats = load_bench_stats(group_dir)
    if not stats:
        raise SystemExit(f"No benchmark sample data found in: {group_dir}")

    render_markdown(
        stats=stats,
        baseline_name=args.baseline,
        output=Path(args.output),
        group=args.group,
        alpha=args.alpha,
        permutation_reps=args.reps,
    )


if __name__ == "__main__":
    main()
