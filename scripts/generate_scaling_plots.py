#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import shutil as _shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class Point:
    backend: str
    n: int
    mean_ns: float
    ci_low_ns: float
    ci_high_ns: float
    std_dev_ns: float


BENCH_RE = re.compile(r"^(cpu|gpu|ndarray) n=(\d+)$")


def format_ns(value: float) -> str:
    if value >= 1_000_000.0:
        return f"{value / 1_000_000.0:.3f} ms"
    if value >= 1_000.0:
        return f"{value / 1_000.0:.3f} µs"
    return f"{value:.3f} ns"


def apply_style() -> None:
    try:
        import scienceplots  # noqa: F401

        if _shutil.which("latex"):
            plt.style.use(["science", "grid", "dark_background"])
        else:
            plt.style.use(["science", "no-latex", "grid", "dark_background"])
    except Exception:
        plt.style.use("dark_background")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "cmr10",
                "Computer Modern Roman",
                "CMU Serif",
                "Latin Modern Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "mathtext.rm": "cmr10",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titleweight": "semibold",
            "axes.labelweight": "medium",
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "figure.facecolor": "#0b1021",
            "axes.facecolor": "#10172a",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#e2e8f0",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "grid.color": "#334155",
            "grid.alpha": 0.7,
            "savefig.facecolor": "#0b1021",
            "savefig.edgecolor": "#0b1021",
            "savefig.transparent": False,
        }
    )


def load_points(group_dir: Path) -> list[Point]:
    points: list[Point] = []
    for entry in sorted(group_dir.iterdir()):
        if not entry.is_dir() or entry.name == "report":
            continue

        match = BENCH_RE.match(entry.name)
        if match is None:
            continue

        backend, n_str = match.groups()
        n = int(n_str)

        estimates_path = entry / "new" / "estimates.json"
        if not estimates_path.exists():
            continue

        estimates = json.loads(estimates_path.read_text(encoding="utf-8"))
        mean = estimates["mean"]["point_estimate"]
        ci_low = estimates["mean"]["confidence_interval"]["lower_bound"]
        ci_high = estimates["mean"]["confidence_interval"]["upper_bound"]
        std_dev = estimates["std_dev"]["point_estimate"]

        points.append(
            Point(
                backend=backend,
                n=n,
                mean_ns=mean,
                ci_low_ns=ci_low,
                ci_high_ns=ci_high,
                std_dev_ns=std_dev,
            )
        )
    return points


def generate_time_plot(points: list[Point], out_path: Path) -> None:
    backend_order = ["cpu", "gpu", "ndarray"]
    markers = {"cpu": "o", "gpu": "s", "ndarray": "^"}
    labels = {"cpu": "CPU", "gpu": "GPU (Metal)", "ndarray": "ndarray"}
    colors = {"cpu": "#5eead4", "gpu": "#fda4af", "ndarray": "#fcd34d"}

    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    xticks = sorted({p.n for p in points})
    for backend in backend_order:
        entries = sorted((p for p in points if p.backend == backend), key=lambda p: p.n)
        if not entries:
            continue
        xs = [p.n for p in entries]
        ys_ms = [p.mean_ns / 1_000_000.0 for p in entries]
        ax.plot(
            xs,
            ys_ms,
            marker=markers[backend],
            markersize=9.0,
            markeredgewidth=0.9,
            linewidth=3.2,
            color=colors[backend],
            label=labels[backend],
        )

    ax.set_xlabel(r"Matrix Size $n$ ($n \times n$)")
    ax.set_ylabel("Mean Runtime (ms)")
    ax.set_title("Matrix Multiplication Scaling")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(v) for v in xticks])
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_speedup_plot(points: list[Point], out_path: Path) -> None:
    cpu_by_n = {p.n: p.mean_ns for p in points if p.backend == "cpu"}
    colors = {"gpu": "#fda4af", "ndarray": "#fcd34d"}

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    xticks = sorted({p.n for p in points})
    for backend, marker, label in [
        ("gpu", "s", "GPU speedup vs CPU"),
        ("ndarray", "^", "ndarray speedup vs CPU"),
    ]:
        entries = sorted(
            (
                p
                for p in points
                if p.backend == backend and p.n in cpu_by_n and p.mean_ns > 0.0
            ),
            key=lambda p: p.n,
        )
        if not entries:
            continue
        xs = [p.n for p in entries]
        ys = [cpu_by_n[p.n] / p.mean_ns for p in entries]
        ax.plot(
            xs,
            ys,
            marker=marker,
            markersize=8.5,
            markeredgewidth=0.8,
            linewidth=3.0,
            color=colors[backend],
            label=label,
        )

    ax.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=1.4)
    ax.set_xlabel(r"Matrix Size $n$ ($n \times n$)")
    ax.set_ylabel(r"Speedup ($\times$)")
    ax.set_title("Relative Speedup vs CPU")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(v) for v in xticks])
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_markdown(
    points: list[Point],
    output_path: Path,
    time_plot_path: Path,
    speedup_plot_path: Path,
) -> None:
    cpu_by_n = {p.n: p.mean_ns for p in points if p.backend == "cpu"}
    ordered = sorted(points, key=lambda p: (p.n, {"cpu": 0, "gpu": 1, "ndarray": 2}[p.backend]))
    rows: list[str] = []
    for p in ordered:
        ci_half = (p.ci_high_ns - p.ci_low_ns) / 2.0
        speed = cpu_by_n[p.n] / p.mean_ns if p.n in cpu_by_n and p.mean_ns > 0 else 0.0
        rows.append(
            "| "
            + " | ".join(
                [
                    str(p.n),
                    p.backend,
                    f"{format_ns(p.mean_ns)} ± {format_ns(ci_half)}",
                    format_ns(p.std_dev_ns),
                    f"{speed:.3f}x",
                ]
            )
            + " |"
        )

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    output_parent = output_path.parent.resolve()
    time_plot_rel = time_plot_path.resolve().relative_to(output_parent)
    speedup_plot_rel = speedup_plot_path.resolve().relative_to(output_parent)

    markdown = "\n".join(
        [
            "# Matrix Scaling Benchmarks",
            "",
            f"- Generated: {ts}",
            "- Bench suite: `matrix_scaling`",
            "- Metric: Criterion mean runtime (plots show mean lines; 95% CI is in the table)",
            "",
            f"![Runtime vs n]({time_plot_rel.as_posix()})",
            "",
            f"![Speedup vs CPU]({speedup_plot_rel.as_posix()})",
            "",
            "| n | backend | μ ± 95% CI | σ | speedup vs cpu |",
            "| ---: | --- | --- | ---: | ---: |",
            *rows,
            "",
            "_Lower runtime is faster._",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scaling benchmarks and generate plots + markdown.")
    parser.add_argument("--criterion-root", default="target/criterion")
    parser.add_argument("--group", default="matrix multiplication scaling")
    parser.add_argument("--output", default="docs/benchmark-scaling.md")
    parser.add_argument("--plot-dir", default="docs/plots")
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()

    criterion_root = Path(args.criterion_root)
    group_dir = criterion_root / args.group

    if not args.no_run:
        if group_dir.exists():
            shutil.rmtree(group_dir)
        subprocess.run(["just", "all"], check=False)
        subprocess.run(["cargo", "bench", "--bench", "matrix_scaling"], check=True)

    if not group_dir.exists():
        raise SystemExit(f"Benchmark group not found: {group_dir}")

    points = load_points(group_dir)
    if not points:
        raise SystemExit(f"No benchmark points found in {group_dir}")

    apply_style()
    plot_dir = Path(args.plot_dir)
    time_plot = plot_dir / "matrix-scaling-time.png"
    speedup_plot = plot_dir / "matrix-scaling-speedup.png"
    generate_time_plot(points, time_plot)
    generate_speedup_plot(points, speedup_plot)
    write_markdown(points, Path(args.output), time_plot, speedup_plot)


if __name__ == "__main__":
    main()
