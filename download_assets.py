#!/usr/bin/env python3
"""Download allowlisted SCOUT runtime assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from scout_router.assets import AssetDownloader, default_asset_specs, named_asset_specs
from scout_router.config import ScoutConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download SCOUT runtime assets.")
    parser.add_argument("--default", action="store_true", help="download predictor, anchor, and fingerprint assets")
    parser.add_argument("--test-data", action="store_true", help="download the SCOUT-450 test dataset")
    parser.add_argument("--cache-dir", type=Path, help="override SCOUT_CACHE_DIR")
    parser.add_argument("--env-file", type=Path, default=THIS_DIR / ".env")
    args = parser.parse_args(argv)

    config = ScoutConfig.from_env(env_path=args.env_file, cache_dir=args.cache_dir)
    specs = []
    if args.default:
        specs.extend(default_asset_specs())
    if args.test_data:
        specs.extend(named_asset_specs(["test_data"]))
    if not specs:
        parser.error("choose at least one of --default or --test-data")

    unique = {spec.name: spec for spec in specs}
    for path in AssetDownloader(config.cache_dir).download(list(unique.values())):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
