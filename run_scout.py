#!/usr/bin/env python3
"""Script entrypoint for SCOUT runtime detection."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from scout_router.config import ScoutConfig
from scout_router.schema import PromptSample

if TYPE_CHECKING:
    from argparse import Namespace

OPTIONAL_HEAVY_DETECTORS = ("d4_attention_tracker", "d5_align_sentinel")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCOUT prompt-injection detection.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="single prompt/content string")
    source.add_argument("--input", type=Path, help="JSONL file with eval_content records")
    parser.add_argument("--output", type=Path, help="optional JSONL output file")
    parser.add_argument(
        "--predictor-input",
        type=Path,
        help="reuse a predictor JSONL file and skip the vLLM predictor phase; only valid with --input",
    )
    parser.add_argument("--details", action="store_true", help="include detector, predictor, and router details")
    parser.add_argument("--cache-dir", type=Path, help="override SCOUT_CACHE_DIR")
    parser.add_argument("--env-file", type=Path, default=THIS_DIR / ".env")
    parser.add_argument(
        "--include-d4",
        action="store_true",
        help="enable D4 attention tracker in the cheap detector pool",
    )
    parser.add_argument(
        "--include-d5",
        action="store_true",
        help="enable D5 alignment sentinel in the cheap detector pool",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="enable all optional heavy cheap detectors: D4 and D5",
    )
    args = parser.parse_args(argv)
    if args.predictor_input is not None and args.input is None:
        parser.error("--predictor-input is only valid with --input")
    return args


def configure_cheap_pool(config: ScoutConfig, args: "Namespace") -> ScoutConfig:
    enabled = list(config.detectors_enabled)
    cheap_pool = list(config.routing.cheap_pool)
    requested = []
    if args.include_heavy or args.include_d4:
        requested.append("d4_attention_tracker")
    if args.include_heavy or args.include_d5:
        requested.append("d5_align_sentinel")

    for detector in requested:
        if detector not in enabled:
            enabled.append(detector)
        if detector not in cheap_pool:
            cheap_pool.append(detector)

    return replace(
        config,
        detectors_enabled=tuple(enabled),
        routing=replace(config.routing, cheap_pool=tuple(cheap_pool)),
    )


def sample_from_record(record: dict, index: int) -> PromptSample:
    if "eval_content" not in record:
        raise ValueError(f"input record {index} is missing eval_content")
    return PromptSample(
        id=str(record.get("id", f"sample-{index:04d}")),
        eval_content=str(record["eval_content"]),
        goal_text=str(record.get("goal_text", "")),
        policy_text=str(record.get("policy_text", "")),
    )


def default_predictor_output_path(input_path: Path, output_path: Path | None) -> Path:
    base_path = output_path or input_path
    suffix = base_path.suffix or ".jsonl"
    return base_path.with_name(f"{base_path.stem}_predictor{suffix}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from scout_router.pipeline import ScoutPipeline

    config = ScoutConfig.from_env(env_path=args.env_file, cache_dir=args.cache_dir)
    config = configure_cheap_pool(config, args)
    def emit(record: dict, handle) -> None:
        line = json.dumps(record, ensure_ascii=False)
        if handle is None:
            print(line)
        else:
            handle.write(line + "\n")

    pipeline = ScoutPipeline(config)
    out_handle = None
    try:
        if args.text is not None:
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                out_handle = args.output.open("w", encoding="utf-8")
            sample = PromptSample(id=None, eval_content=args.text)
            emit(pipeline.detect(sample, details=args.details), out_handle)
            return 0

        predictor_path = args.predictor_input or default_predictor_output_path(args.input, args.output)
        should_generate_predictor = args.predictor_input is None and not predictor_path.exists()
        if should_generate_predictor:
            predictor_path.parent.mkdir(parents=True, exist_ok=True)
            pred_handle = predictor_path.open("w", encoding="utf-8")
            try:
                with args.input.open(encoding="utf-8") as handle:
                    for index, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        sample = sample_from_record(json.loads(line), index)
                        emit(pipeline.build_prediction_record(sample), pred_handle)
            finally:
                pred_handle.close()
                pipeline.close_predictor()

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            out_handle = args.output.open("w", encoding="utf-8")
        with predictor_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                emit(pipeline.detect_from_prediction_record(json.loads(line), details=args.details), out_handle)
    finally:
        if out_handle is not None:
            out_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
