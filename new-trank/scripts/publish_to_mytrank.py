#!/usr/bin/env python3
"""
publish_to_mytrank.py

Explicit publish step:
- Copies `new-trank/exports/*` into `my-trank/public/data/*`.
- This is the ONLY intended pathway for the new-trank pipeline to touch the site.

Safety:
- Does not read or write anything in `ml model/`.
"""

from __future__ import annotations

import pathlib
import shutil


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "new-trank" / "exports"
DST = REPO_ROOT / "my-trank" / "public" / "data"


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"[error] missing exports dir: {SRC}")

    DST.mkdir(parents=True, exist_ok=True)

    copied = 0
    for p in SRC.glob("*"):
        if p.is_dir():
            # Allow nested assets later (e.g., international history subsets)
            shutil.copytree(p, DST / p.name, dirs_exist_ok=True)
            copied += 1
            continue
        shutil.copy2(p, DST / p.name)
        copied += 1

    print(f"[ok] published {copied} artifact(s) to {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

