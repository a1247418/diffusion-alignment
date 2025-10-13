import os
import sys
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def load_alignment_results(alignment_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load nested alignment results dict: source -> model -> module -> acc.

    Handles both pickle.dump files and numpy-saved pickled objects.
    """
    if not alignment_path.exists():
        return {}

    # Try pickle first
    try:
        with open(alignment_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback: numpy load (if saved via np.save with allow_pickle)
    try:
        arr = np.load(str(alignment_path), allow_pickle=True)
        # np.load might return an ndarray holding a dict-like object
        if hasattr(arr, "item"):
            data = arr.item()
        else:
            data = arr
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {}


def discover_data_roots(repo_root: Path) -> List[Tuple[str, Path]]:
    """Discover THINGS data roots, including distance variants.

    Returns list of (distance_name, path) where distance_name is e.g. "cosine", "euclidean", "dot".
    """
    base = repo_root / "things_playground"
    candidates = list(base.glob("things_data*"))
    results: List[Tuple[str, Path]] = []
    for p in sorted(candidates):
        if not p.is_dir():
            continue
        # Infer distance by suffix after first underscore; default to cosine when no suffix
        name = p.name
        if name == "things_data":
            dist = "cosine"
        elif name.startswith("things_data_"):
            dist = name.split("_", 2)[-1]
        else:
            dist = name
        results.append((dist, p))
    return results


def format_alignment_section(repo_root: Path) -> str:
    lines: List[str] = []
    lines.append("== Alignment Results (odd-one-out accuracy) ==")
    data_roots = discover_data_roots(repo_root)
    if not data_roots:
        lines.append("No THINGS data roots found under things_playground/.")
        return "\n".join(lines)

    for dist, root in data_roots:
        alignment_file = root / "alignment.pkl"
        results = load_alignment_results(alignment_file)
        lines.append(f"\n[distance={dist}] file={alignment_file}")
        if not results:
            lines.append("  (no alignment results found)")
            continue

        # Collect rows: (source, model, module, acc)
        rows: List[Tuple[str, str, str, float]] = []
        for source, by_model in results.items():
            if not isinstance(by_model, dict):
                continue
            for model, by_module in by_model.items():
                if not isinstance(by_module, dict):
                    continue
                for module, acc in by_module.items():
                    try:
                        acc_f = float(acc)
                    except Exception:
                        # Some files may store additional structures; skip if not a scalar
                        continue
                    rows.append((source, model, module, acc_f))

        if not rows:
            lines.append("  (no scalar accuracies found)")
            continue

        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        lines.append("  source | model | module | acc")
        for source, model, module, acc in rows:
            lines.append(f"  {source} | {model} | {module} | {acc:.4f}")

    return "\n".join(lines)


def load_probing_results(repo_root: Path) -> pd.DataFrame:
    results_path = repo_root / "things_playground" / "results" / "probing_results.pkl"
    if not results_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_pickle(results_path)
    except Exception:
        return pd.DataFrame()


def format_probing_section(repo_root: Path) -> str:
    lines: List[str] = []
    lines.append("== Probing Results (k-fold validation) ==")
    df = load_probing_results(repo_root)
    if df.empty:
        lines.append("No probing results found under things_playground/results/probing_results.pkl.")
        return "\n".join(lines)

    # Normalize column names if necessary
    cols_map = {c.lower(): c for c in df.columns}
    def get(col: str) -> str:
        return cols_map.get(col, col)

    # Expected columns per main_probing.py
    col_model = get("model")
    col_module = get("module")
    col_source = get("source")
    col_l2 = get("l2_reg")
    col_optim = get("optim")
    col_lr = get("lr")
    col_folds = get("n_folds")
    col_bias = get("bias")
    col_acc = get("probing")
    col_loss = get("cross-entropy")

    # Sort for readability
    sort_cols = [c for c in [col_source, col_model, col_module, col_folds, col_l2, col_optim, col_lr] if c in df.columns]
    try:
        df_sorted = df.sort_values(by=sort_cols)
    except Exception:
        df_sorted = df

    lines.append("source | model | module | folds | l2 | optim | lr | bias | probing_acc | xent_loss")
    for _, row in df_sorted.iterrows():
        src = row.get(col_source, "?")
        mdl = row.get(col_model, "?")
        mod = row.get(col_module, "?")
        folds = row.get(col_folds, "?")
        l2 = row.get(col_l2, "?")
        optim = row.get(col_optim, "?")
        lr = row.get(col_lr, "?")
        bias = row.get(col_bias, "?")
        acc = row.get(col_acc, "?")
        loss = row.get(col_loss, "?")
        try:
            acc = f"{float(acc):.4f}"
        except Exception:
            acc = str(acc)
        try:
            loss = f"{float(loss):.4f}"
        except Exception:
            loss = str(loss)
        lines.append(f"{src} | {mdl} | {mod} | {folds} | {l2} | {optim} | {lr} | {bias} | {acc} | {loss}")

    return "\n".join(lines)


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    out_path = repo_root / "easy_results.txt"

    sections: List[str] = []
    sections.append(format_alignment_section(repo_root))
    sections.append("")
    sections.append(format_probing_section(repo_root))
    text = "\n".join(sections).strip() + "\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wrote results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

