#!/usr/bin/env python3
"""
Validate critical local file references in key docs.

Default docs:
  - docs/INDEX.md
  - docs/missing_data_closure_runbook.md
  - PROJECT_MAP.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


BACKTICK_RE = re.compile(r"`([^`]+)`")
MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def looks_like_local_file_ref(token: str) -> bool:
    t = token.strip()
    if not t or t.startswith(("http://", "https://", "mailto:", "#")):
        return False
    if any(ch.isspace() for ch in t):
        return False
    if any(x in t for x in ["{", "}", "*", "<", ">"]):
        return False
    if t.startswith(("/", "~")):
        return False
    if not re.match(r"^[A-Za-z0-9_./-]+$", t):
        return False
    if "/" not in t:
        return False
    # Validate only structural source links, not generated runtime artifacts.
    allowed_ext = {".md", ".py", ".sql", ".sh", ".json", ".yaml", ".yml"}
    return Path(t).suffix in allowed_ext


def resolve_ref(doc_path: Path, ref: str, repo_root: Path) -> Path | None:
    ref = ref.strip()
    candidates = [doc_path.parent / ref, repo_root / ref, repo_root / "college_scripts" / ref]
    for c in candidates:
        if c.exists():
            return c
    return None


def extract_refs(text: str) -> list[str]:
    refs = []
    refs.extend([m.group(1).strip() for m in BACKTICK_RE.finditer(text)])
    refs.extend([m.group(1).strip() for m in MD_LINK_RE.finditer(text)])
    out = []
    for r in refs:
        if looks_like_local_file_ref(r):
            out.append(r)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Check critical markdown file references.")
    parser.add_argument(
        "--docs",
        nargs="*",
        default=["docs/INDEX.md", "docs/missing_data_closure_runbook.md", "PROJECT_MAP.md"],
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    missing = []

    for rel in args.docs:
        doc = (repo_root / rel).resolve()
        if not doc.exists():
            missing.append((rel, "<doc_missing>"))
            continue
        text = doc.read_text(encoding="utf-8", errors="ignore")
        for ref in extract_refs(text):
            if resolve_ref(doc, ref, repo_root) is None:
                missing.append((str(doc.relative_to(repo_root)), ref))

    if missing:
        print("Doc health check failed. Missing critical references:")
        for doc, ref in missing:
            print(f"- {doc} :: {ref}")
        return 1

    print("Doc health check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
