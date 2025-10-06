#!/usr/bin/env python3
"""Run debug recipes across residual connection method/pattern combinations.

This helper is intended to be executed **after** logging into the target node
and activating the desired Python environment.  It iterates over the debug
recipes (default: ``recipes/debug*.yaml``), sweeps through residual connection
methods (linear / orthogonal) and residual patterns (default, rezero,
rezero_constrained, rescale_stream), and launches ``train_classifier.py`` runs
with a very small number of steps to validate that activation metrics are
recorded successfully.

Each run writes into a dedicated results directory (default:
``results-classifier-debug``).  After completion the script scans the produced
``log.txt`` file to assert that activation metrics (``activation/`` entries)
were emitted.  Any missing metric log causes the script to abort.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml


DEFAULT_PATTERNS: Sequence[str] = (
    "default",
    "rezero",
    "rezero_constrained",
    "rescale_stream",
)
DEFAULT_METHODS: Sequence[str] = ("linear", "orthogonal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipes",
        type=str,
        nargs="*",
        default=None,
        help="Explicit list of recipe files to run. When omitted all recipes/debug*.yaml files are used.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results-classifier-debug"),
        help="Root directory where experiment sub-folders will be created.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=10,
        help="Number of optimizer steps per run (keep small for fast validation).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs per run (used together with max-train-steps to limit runtime).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Logging interval passed to train_classifier (ensures metrics within few steps).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        choices=DEFAULT_METHODS,
        help="Subset of residual methods to evaluate.",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="*",
        default=None,
        choices=DEFAULT_PATTERNS,
        help="Subset of residual patterns to evaluate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--torchrun-binary",
        type=str,
        default="torchrun",
        help="torchrun executable to launch distributed training.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of processes per node to launch via torchrun.",
    )
    return parser.parse_args()


def discover_recipes(recipe_args: Sequence[str] | None) -> List[Path]:
    if recipe_args:
        return [Path(path).resolve() for path in recipe_args]
    return sorted(Path("recipes").glob("debug*.yaml"))


def load_model_name(recipe_path: Path) -> str:
    with recipe_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return str(data.get("model", "")).lower()


def determine_rescale_modes(model_name: str) -> Sequence[str]:
    if model_name.startswith("resnet"):
        return ("scalar", "conv1x1")
    return ("scalar",)


def build_command(
    torchrun_bin: str,
    recipe: Path,
    method: str,
    pattern: str,
    rescale_mode: str | None,
    results_root: Path,
    max_train_steps: int,
    epochs: int,
    log_interval: int,
    nproc_per_node: int,
) -> List[str]:
    cmd: List[str] = [
        torchrun_bin,
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "train_classifier.py",
        "--config_file",
        str(recipe),
        "--results_dir",
        str(results_root),
        "--epochs",
        str(epochs),
        "--max_train_steps",
        str(max_train_steps),
        "--log_interval",
        str(log_interval),
        "--save_every_steps",
        str(max_train_steps * 1000),
        "--save_every_epochs",
        str(epochs * 1000),
        "--debug",
    ]
    if method == "orthogonal":
        cmd.append("--orthogonal_residual")
    else:
        cmd.append("--no_orthogonal_residual")
    if pattern != "default":
        cmd.extend(["--residual_pattern", pattern])
    if pattern == "rescale_stream" and rescale_mode is not None:
        cmd.extend(["--residual_rescale_mode", rescale_mode])
    return cmd


def command_to_str(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def capture_existing(result_root: Path) -> set[Path]:
    if not result_root.exists():
        return set()
    return {path for path in result_root.iterdir() if path.is_dir()}


def ensure_activation_metrics(result_root: Path, before: Iterable[Path]) -> None:
    before_set = set(before)
    after_set = capture_existing(result_root)
    new_dirs = sorted(after_set - before_set)
    if not new_dirs:
        raise RuntimeError("No new results directory created; cannot verify metrics")

    activation_found = False
    for run_dir in reversed(new_dirs):  # newest last
        log_path = run_dir / "log.txt"
        if not log_path.exists():
            continue
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "activation/" in line:
                    activation_found = True
                    break
        if activation_found:
            break
    if not activation_found:
        raise RuntimeError(
            f"Activation metrics not detected in logs for runs under {result_root}. "
            "Inspect log files for details."
        )


def main() -> None:
    args = parse_args()
    recipes = discover_recipes(args.recipes)
    if not recipes:
        raise SystemExit("No recipes found. Provide --recipes or populate recipes/debug*.yaml.")

    methods = args.methods or DEFAULT_METHODS
    patterns = args.patterns or DEFAULT_PATTERNS

    args.results_root.mkdir(parents=True, exist_ok=True)

    for recipe in recipes:
        model_name = load_model_name(recipe)
        rescale_modes = determine_rescale_modes(model_name)

        for method, pattern in itertools.product(methods, patterns):
            mode_iter: Sequence[str | None]
            if pattern == "rescale_stream":
                mode_iter = rescale_modes
            else:
                mode_iter = (None,)

            for mode in mode_iter:
                cmd = build_command(
                    args.torchrun_binary,
                    recipe,
                    method,
                    pattern,
                    mode,
                    args.results_root,
                    args.max_train_steps,
                    args.epochs,
                    args.log_interval,
                    args.nproc_per_node,
                )
                print(command_to_str(cmd))
                if args.dry_run:
                    continue

                before = capture_existing(args.results_root)
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if proc.returncode != 0:
                    print(proc.stdout)
                    raise RuntimeError(f"Command failed with exit code {proc.returncode}")
                # Keep logs for optional post-mortem.
                sys.stdout.write(proc.stdout)
                sys.stdout.flush()

                ensure_activation_metrics(args.results_root, before)


if __name__ == "__main__":  # pragma: no cover
    main()
