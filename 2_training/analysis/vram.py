# quick chatgpt script (I checked it for correctness) to analyze the vram logs produced when model_config.will_log_vram == True
# simply to help find the source of the oom issues I'm having

import pandas as pd

def vram_diag(log_path, top=20, min_count=1):
    # --- Load ---
    df = pd.read_json(log_path, lines=True)

    # --- Normalize / guard ---
    numeric_cols = [
        "alloc_mb", "reserved_mb", "peak_mb",
        "delta_alloc_mb", "delta_reserved_mb", "delta_peak_mb",
        "is_oom"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["tag"] = df["tag"].fillna("UNKNOWN")
    df["step"] = df["step"].fillna(-1).astype(int)
    df["epoch"] = df["epoch"].fillna(-1).astype(int)
 
    print(f"\nLoaded {len(df)} rows")
    print(f"OOM rows: {int(df['is_oom'].sum())}")
    print("-" * 80)

    # ==========================================================================
    # 1) Per-tag problem summary
    # ==========================================================================
    tag_summary = (
        df.groupby("tag")
        .agg(
            count=("tag", "size"),
            ooms=("is_oom", "sum"),
            peak_max=("peak_mb", "max"),
            peak_p50=("peak_mb", "median"),
            peak_mean=("peak_mb", "mean"),
            delta_alloc_max=("delta_alloc_mb", "max"),
            delta_alloc_p50=("delta_alloc_mb", "median"),
        )
        .query("count >= @min_count")
        .sort_values(
            by=["ooms", "peak_max", "delta_alloc_max"],
            ascending=False
        )
    )

    print("=== Per-tag problem summary ===")
    print(tag_summary.head(top).round(2))
    print("-" * 80)

    # ==========================================================================
    # 2) Worst individual steps by PEAK VRAM
    # ==========================================================================
    worst_peak = (
        df.sort_values(
            by=["is_oom", "peak_mb"],
            ascending=[False, False]
        )
        .loc[:, [
            "epoch", "step", "tag", "is_oom",
            "peak_mb", "delta_alloc_mb",
            "alloc_mb", "reserved_mb"
        ]]
    )

    print(f"=== Top {top} worst steps by PEAK VRAM ===")
    print(worst_peak.head(top).round(2))
    print("-" * 80)

    # ==========================================================================
    # 3) Worst individual steps by ALLOCATION SPIKE
    # ==========================================================================
    worst_delta = (
        df.sort_values(
            by=["is_oom", "delta_alloc_mb"],
            ascending=[False, False]
        )
        .loc[:, [
            "epoch", "step", "tag", "is_oom",
            "delta_alloc_mb", "peak_mb",
            "alloc_mb", "reserved_mb"
        ]]
    )

    print(f"=== Top {top} worst steps by ΔALLOC ===")
    print(worst_delta.head(top).round(2))
    print("-" * 80)

    # ==========================================================================
    # 4) OOM-only diagnostics (usually the most actionable)
    # ==========================================================================
    if df["is_oom"].any():
        oom_df = df[df["is_oom"] == 1].copy()

        print("=== OOM rows (first few) ===")
        cols = ["epoch", "step", "tag", "peak_mb", "delta_alloc_mb", "error"]
        print(oom_df[cols].head(top).round(2))
        print("-" * 80)