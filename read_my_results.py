#!/usr/bin/env python3
"""
Summarize selected accuracy metric from a folder of files like:
  t10_block3.txt  -> contains lines like:
    "Zero-shot odd-one-out accuracy (t=10%): 0.3860"
    "Probing transformed odd-one-out accuracy: 0.4832"
    "Probing cross-validated accuracy (fold-wise OOO): 0.4474"

Creates:
  1) a Markdown file with the folder name as a title and a table of accuracies
  2) a CSV with the same base name as the Markdown output ('.csv' appended)

Usage:
  python summarize_accuracies.py \
    --folder sd3_results/label \
    --out sd3_results_label.md \
    --metric zero-shot|probing-transformed|probing-cv-ooo
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

FILENAME_RE = re.compile(r"^t(?P<noise>\d+)_block(?P<block>\d+)\.txt$", re.IGNORECASE)

# Float pattern
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")

# Metric-specific patterns (capture the numeric value)
# We allow optional "(t=...)" on the zero-shot line so it works across t values.
METRIC_PATTERNS = {
    "zero-shot": re.compile(
        r"Zero-shot\s+odd-one-out\s+accuracy(?:\s*\(t\s*=\s*\d+%?\))?\s*:\s*([-+]?(?:\d*\.\d+|\d+))",
        re.IGNORECASE,
    ),
    "probing-transformed": re.compile(
        r"Probing\s+transformed\s+odd-one-out\s+accuracy\s*:\s*([-+]?(?:\d*\.\d+|\d+))",
        re.IGNORECASE,
    ),
    "probing-cv-ooo": re.compile(
        r"Probing\s+cross-validated\s+accuracy\s*\(fold-wise\s+OOO\)\s*:\s*([-+]?(?:\d*\.\d+|\d+))",
        re.IGNORECASE,
    ),
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize accuracies into Markdown + CSV.")
    p.add_argument("--folder", type=Path, default="sd3_results/probing_label")
    p.add_argument("--out", type=Path, default="sd3_results_probing_label.md")
    p.add_argument(
        "--metric",
        type=str,
        choices=["zero-shot", "probing-transformed", "probing-cv-ooo"],
        default="probing-cv-ooo",
    )
    return p.parse_args()

def read_value(file_path: Path, metric: str) -> Optional[float]:
    """
    Reads the requested accuracy value from file content.

    Only the selected metric is considered. If the selected metric
    is not present in the file, returns None.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    pat = METRIC_PATTERNS.get(metric)
    if not pat:
        # Should not happen because argparse limits choices, but keep a guard.
        raise ValueError(f"Unknown metric: {metric}")

    m = pat.search(text)
    if not m:
        return None

    try:
        return float(m.group(1))
    except ValueError:
        return None

def collect(folder: Path, metric: str) -> Tuple[Dict[int, Dict[int, float]], Set[int], Set[int]]:
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
        val = read_value(entry, metric)
        if val is None:
            continue
        blocks.add(block)
        noises.add(noise)
        table.setdefault(block, {})[noise] = val

    return table, blocks, noises

def write_markdown(md_path: Path, folder: Path, metric: str,
                   table: Dict[int, Dict[int, float]], blocks: Set[int], noises: Set[int]) -> None:
    noises_sorted = sorted(noises)
    blocks_sorted = sorted(blocks)

    lines = []
    lines.append(f"# {folder.name}")
    lines.append("")
    lines.append(f"**Metric:** {metric}")
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

def write_csv(csv_path: Path, metric: str,
              table: Dict[int, Dict[int, float]], blocks: Set[int], noises: Set[int]) -> None:
    noises_sorted = sorted(noises)
    blocks_sorted = sorted(blocks)

    out_lines = []
    # Include the metric on a commented header line for traceability
    out_lines.append(f"# metric,{metric}")
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
    metric: str = args.metric
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {folder}")

    table, blocks, noises = collect(folder, metric)
    if not blocks or not noises:
        # Explicit error message mentioning the chosen metric
        raise SystemExit(
            "No matching files/values found. "
            f"Expected names like t10_block3.txt containing the selected metric '{metric}'."
        )

    # Markdown path is exactly what the user passed.
    md_path: Path = args.out

    # CSV path: same base name, with '.csv' appended (so 'results.md.csv' if you pass 'results.md').
    csv_path: Path = Path(str(md_path))  # copy
    csv_path = csv_path.with_name(csv_path.name + ".csv")

    write_markdown(md_path, folder, metric, table, blocks, noises)
    write_csv(csv_path, metric, table, blocks, noises)

    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote CSV:      {csv_path}")

if __name__ == "__main__":
    main()
 