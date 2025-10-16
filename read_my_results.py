#!/usr/bin/env python3
"""
Summarize block/noise accuracies from a folder of files like:
  t10_block3.txt  -> contains a line like:
  "Zero-shot odd-one-out accuracy (t=10%): 0.3851"

Creates:
  1) a Markdown file with the folder name as a title and a table of accuracies
  2) a CSV with the same base name as the Markdown output ('.csv' appended)

Usage:
  python summarize_accuracies.py --folder sd3_results/label --out sd3_results_label.md
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Set, Tuple

FILENAME_RE = re.compile(r"^t(?P<noise>\d+)_block(?P<block>\d+)\.txt$", re.IGNORECASE)

# Float pattern
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")

# Prefer the number after the word 'accuracy' and a colon, e.g. "accuracy ... : 0.3851"
ACCURACY_AFTER_COLON_RE = re.compile(
    r"accuracy.*?:\s*([-+]?(?:\d*\.\d+|\d+))",
    re.IGNORECASE | re.DOTALL
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize accuracies into Markdown + CSV.")
    p.add_argument("--folder", type=Path, default="sd3_results")
    p.add_argument("--out", type=Path, default="sd3_results.md")
    return p.parse_args()

def read_value(file_path: Path) -> float | None:
    """
    Reads the accuracy value from file content like:
    "Zero-shot odd-one-out accuracy (t=10%): 0.3851"

    Strategy:
      1) Try to match the float immediately after 'accuracy ... :'
      2) Otherwise, take the LAST float in the file (so '0.3851', not the '10' from 't=10%')
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # 1) Prefer explicit "accuracy ... : <float>"
    m = ACCURACY_AFTER_COLON_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass  # fall back below

    # 2) Fallback: take the last float in the file
    floats = FLOAT_RE.findall(text)
    if not floats:
        return None
    try:
        return float(floats[-1])
    except ValueError:
        return None

def collect(folder: Path) -> Tuple[Dict[int, Dict[int, float]], Set[int], Set[int]]:
    """
    Returns:
      table[block_id][noise] = value
      blocks: set of block ids
      noises: set of noise levels
    """
    table: Dict[int, Dict[int, float]] = {}
    blocks: Set[int] = set()
    noises: Set[int] = set()

    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        m = FILENAME_RE.match(entry.name)
        if not m:
            continue
        noise = int(m.group("noise"))
        block = int(m.group("block"))
        val = read_value(entry)
        if val is None:
            continue
        blocks.add(block)
        noises.add(noise)
        table.setdefault(block, {})[noise] = val

    return table, blocks, noises

def write_markdown(md_path: Path, folder: Path, table: Dict[int, Dict[int, float]], blocks: Set[int], noises: Set[int]) -> None:
    noises_sorted = sorted(noises)
    blocks_sorted = sorted(blocks)

    lines = []
    lines.append(f"# {folder.name}")
    lines.append("")
    # header
    header = ["block \\ noise"] + [f"t{n}" for n in noises_sorted]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    # rows
    for b in blocks_sorted:
        row = [f"block{b}"]
        for n in noises_sorted:
            v = table.get(b, {}).get(n, None)
            row.append("" if v is None else f"{v:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_csv(csv_path: Path, table: Dict[int, Dict[int, float]], blocks: Set[int], noises: Set[int]) -> None:
    noises_sorted = sorted(noises)
    blocks_sorted = sorted(blocks)

    out_lines = []
    header = ["block"] + [f"t{n}" for n in noises_sorted]
    out_lines.append(",".join(header))
    for b in blocks_sorted:
        row = [str(b)]
        for n in noises_sorted:
            v = table.get(b, {}).get(n, None)
            row.append("" if v is None else f"{v:.6f}")
        out_lines.append(",".join(row))

    csv_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

def main():
    args = parse_args()
    folder: Path = args.folder
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {folder}")

    table, blocks, noises = collect(folder)
    if not blocks or not noises:
        raise SystemExit("No matching files found (expected names like t10_block3.txt with '... accuracy (t=10%): 0.3851').")

    # Markdown path is exactly what the user passed.
    md_path: Path = args.out

    # CSV path: same base name, with '.csv' appended (so 'results.md.csv' if you pass 'results.md').
    csv_path: Path = Path(str(md_path))  # copy
    csv_path = csv_path.with_name(csv_path.name + ".csv")

    write_markdown(md_path, folder, table, blocks, noises)
    write_csv(csv_path, table, blocks, noises)

    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote CSV:      {csv_path}")

if __name__ == "__main__":
    main()
