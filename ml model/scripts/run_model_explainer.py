#!/usr/bin/env python3
"""
Stack 2026 Interpretability Tool (2026).

1. Latent space clustering (KMeans k=5) for global archetype drivers.
2. Gradient-based attribution for specific players (Ayton, Bagley, Shai).

Outputs: terminal report + optional HTML to data/audit/stack2026_explainer_report.html
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from train_2026_model import Stack2026Model  # noqa: E402

SUP_PATH = BASE / "data" / "training" / "unified_training_table_supervised.parquet"
CKPT_PATH = BASE / "models" / "stack2026" / "stack2026_supervised.pt"
AUDIT_DIR = BASE / "data" / "audit"
N_CLUSTERS = 5
TARGET_PLAYERS = ["Deandre Ayton", "Marvin Bagley III", "Shai Gilgeous-Alexander"]


def _as_numeric_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    frame = df[[c for c in cols if c in df.columns]].copy()
    for c in frame.columns:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    # ensure all cols present
    for c in cols:
        if c not in frame.columns:
            frame[c] = 0.0
    return frame[cols].fillna(0.0).to_numpy(dtype=np.float32)


def _extract_context(df: pd.DataFrame, context_cols: list) -> np.ndarray:
    if not context_cols:
        return np.zeros((len(df), 3), dtype=np.float32)
    cols = [c for c in context_cols if c in df.columns]
    if not cols:
        return np.zeros((len(df), 3), dtype=np.float32)
    frame = df[cols].copy()
    for c in cols:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    ctx = frame.fillna(0.0).to_numpy(dtype=np.float32)
    if ctx.shape[1] < 3:
        ctx = np.pad(ctx, ((0, 0), (0, 3 - ctx.shape[1])))
    return ctx


def main() -> None:
    print("=" * 60)
    print("Stack 2026 Model Explainer (Latent Clustering + Gradient Attribution)")
    print("=" * 60)

    if not SUP_PATH.exists():
        print(f"ERROR: Supervised table not found: {SUP_PATH}")
        sys.exit(1)
    if not CKPT_PATH.exists():
        print(f"ERROR: Checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    # ----- STEP 1: Load data and model -----
    df = pd.read_parquet(SUP_PATH)
    # Valid, NBA-mapped: has nba_id
    if "nba_id" in df.columns:
        df = df[pd.to_numeric(df["nba_id"], errors="coerce").notna()].copy()
    if "player_name" not in df.columns and "norm_name" in df.columns:
        df["player_name"] = df["norm_name"]
    if "player_name" not in df.columns:
        df["player_name"] = "unknown"
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} NBA-mapped rows from supervised table.")

    ckpt = torch.load(CKPT_PATH, weights_only=False)
    feature_cols = ckpt.get("feature_cols")
    if not feature_cols:
        print("ERROR: Checkpoint missing feature_cols")
        sys.exit(1)
    means = np.array(ckpt["means"], dtype=np.float32)
    stds = np.array(ckpt["stds"], dtype=np.float32)
    context_cols = ckpt.get("context_cols", [])

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = _as_numeric_matrix(df, feature_cols)
    X_norm = (X - means) / np.where(stds > 1e-9, stds, 1.0)
    ctx = _extract_context(df, context_cols)

    input_dim = X_norm.shape[1]
    use_hyper = len(context_cols) > 0
    n_targets = int(ckpt.get("n_targets", 1))
    model = Stack2026Model(input_dim=input_dim, use_hypernetwork=use_hyper, n_targets=max(1, n_targets))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ----- STEP 2: Extract latent z and cluster -----
    x_t = torch.FloatTensor(X_norm)
    with torch.no_grad():
        z = model.autoencoder.encode(x_t)
    z_np = z.detach().numpy()

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_id = kmeans.fit_predict(z_np)
    df["cluster_id"] = cluster_id
    print(f"Clustering: k={N_CLUSTERS}, latent shape={z_np.shape}")

    # ----- STEP 3: Full gradient attribution (per-sample to avoid graph issues) -----
    x_grads = np.zeros_like(X_norm, dtype=np.float32)
    ctx_grads = np.zeros_like(ctx, dtype=np.float32)

    for i in range(len(df)):
        xi = torch.FloatTensor(X_norm[i : i + 1]).clone().requires_grad_(True)
        ci = torch.FloatTensor(ctx[i : i + 1]).clone().requires_grad_(True)
        mu, logvar = model(xi, ci)
        # scalar output for backward: sum of predictions (same as gradient flow for ranking)
        out = mu.sum()
        out.backward()
        if xi.grad is not None:
            x_grads[i] = xi.grad.detach().numpy().flatten()
        if ci.grad is not None:
            ctx_grads[i] = ci.grad.detach().numpy().flatten()
        # clear for next iteration
        model.zero_grad()

    # ----- STEP 4: Macro (archetypes) -----
    print("\n" + "=" * 60)
    print("MACRO: Archetypes (Top players per cluster + Top 5 driving features)")
    print("=" * 60)

    target_col = ckpt.get("target_col", "y_peak_epm_window")
    pred_mu_col = "pred_mu_explainer"
    with torch.no_grad():
        mu_all, _ = model(torch.FloatTensor(X_norm), torch.FloatTensor(ctx))
    # Primary (first head) for ranking
    mu_primary = mu_all[:, 0].numpy() if mu_all.dim() > 1 and mu_all.shape[1] > 1 else mu_all.numpy().flatten()
    df[pred_mu_col] = mu_primary

    for cid in range(N_CLUSTERS):
        mask = df["cluster_id"] == cid
        sub = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        # Top 3-4 players by predicted score for human labeling
        top = sub.nlargest(4, pred_mu_col)
        names = top["player_name"].astype(str).tolist()
        print(f"\n--- Cluster {cid} ---")
        print("  Representative players (for labeling):", ", ".join(names))

        # Mean absolute gradient per feature in this cluster
        mag = np.abs(x_grads[mask.values]).mean(axis=0)
        order = np.argsort(mag)[::-1]
        top5_idx = order[:5]
        top5_names = [feature_cols[j] for j in top5_idx]
        print("  Top 5 driving features (mean |grad|):", top5_names)

    # ----- STEP 5: Micro (target players) -----
    print("\n" + "=" * 60)
    print("MICRO: Target players (pred, cluster, top +/- drivers)")
    print("=" * 60)

    for name_part in TARGET_PLAYERS:
        # match substring (e.g. "Shai" in "Shai Gilgeous-Alexander")
        mask = df["player_name"].astype(str).str.contains(name_part.split()[0], case=False, na=False)
        if not mask.any():
            # try full name
            mask = df["player_name"].astype(str).str.contains(name_part, case=False, na=False)
        if not mask.any():
            print(f"\n  {name_part}: NOT FOUND in supervised table")
            continue
        idx = mask.idxmax()  # first match
        row = df.loc[idx]
        cid = int(row["cluster_id"])
        pred = float(row[pred_mu_col])
        print(f"\n  {row['player_name']} (cluster {cid}, pred={pred:.4f})")

        g = x_grads[df.index.get_loc(idx)]
        order_pos = np.argsort(g)[::-1]
        order_neg = np.argsort(g)[:5]
        top5_pos = [(feature_cols[i], g[i], X[df.index.get_loc(idx), i]) for i in order_pos[:5]]
        top5_neg = [(feature_cols[i], g[i], X[df.index.get_loc(idx), i]) for i in order_neg[:5]]
        print("  Top 5 positive drivers (grad, raw_val):")
        for fn, gr, rv in top5_pos:
            print(f"    {fn}: grad={gr:.4f} raw={rv:.4f}")
        print("  Top 5 negative drivers (grad, raw_val):")
        for fn, gr, rv in top5_neg:
            print(f"    {fn}: grad={gr:.4f} raw={rv:.4f}")

    # ----- STEP 6: Optional HTML report -----
    html_path = AUDIT_DIR / "stack2026_explainer_report.html"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Stack 2026 Explainer</title>")
        f.write("<style>body{font-family:sans-serif;margin:1rem;} table{border-collapse:collapse;} th,td{border:1px solid #ccc;padding:4px 8px;} pre{background:#f5f5f5;padding:8px;}</style></head><body>")
        f.write("<h1>Stack 2026 Model Explainer Report</h1>")
        f.write("<h2>Archetypes (by cluster)</h2>")
        for cid in range(N_CLUSTERS):
            mask = df["cluster_id"] == cid
            sub = df.loc[mask]
            if len(sub) == 0:
                continue
            top = sub.nlargest(4, pred_mu_col)
            names = ", ".join(top["player_name"].astype(str))
            mag = np.abs(x_grads[mask.values]).mean(axis=0)
            order = np.argsort(mag)[::-1][:5]
            feats = ", ".join([feature_cols[i] for i in order])
            f.write(f"<p><b>Cluster {cid}</b>: {names}. Top features: {feats}</p>")
        f.write("<h2>Target players</h2>")
        for name_part in TARGET_PLAYERS:
            mask = df["player_name"].astype(str).str.contains(name_part.split()[0], case=False, na=False)
            if not mask.any():
                f.write(f"<p><b>{name_part}</b>: not found.</p>")
                continue
            idx = mask.idxmax()
            row = df.loc[idx]
            g = x_grads[df.index.get_loc(idx)]
            pos = ", ".join([f"{feature_cols[i]}({g[i]:.3f})" for i in np.argsort(g)[::-1][:5]])
            neg = ", ".join([f"{feature_cols[i]}({g[i]:.3f})" for i in np.argsort(g)[:5]])
            f.write(f"<p><b>{row['player_name']}</b> cluster={int(row['cluster_id'])}, pred={row[pred_mu_col]:.4f}. ")
            f.write(f"Positive: {pos}. Negative: {neg}</p>")
        f.write("</body></html>")
    print(f"\nWrote HTML report: {html_path}")

    print("\n" + "=" * 60)
    print("To re-run: cd \"ml model\" && python3 scripts/run_model_explainer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
