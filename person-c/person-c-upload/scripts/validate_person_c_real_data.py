"""Read-only validation of Person C FC targets on the real training split."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baseline.evaluation import build_matched_control_pairs, compute_fold_change


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--missing-threshold", type=float, default=0.80)
    args = parser.parse_args()

    metadata_path = args.data_dir / "WAYB_WAYC_metadata_train_val(1).csv"
    proteome_path = args.data_dir / "WAYB_WAYC_proteome_raw_train_val.csv"

    started = time.perf_counter()
    meta = pd.read_csv(metadata_path).set_index("sample_ID")
    protein_columns = pd.read_csv(proteome_path, nrows=0).columns.tolist()
    proteome_dtypes = {column: np.float32 for column in protein_columns if column != "sample_ID"}
    proteome = pd.read_csv(proteome_path, dtype=proteome_dtypes).set_index("sample_ID")
    print(f"读取训练/验证原始文件: {time.perf_counter() - started:.1f}s", flush=True)
    common_ids = meta.index.intersection(proteome.index, sort=False)
    meta = meta.loc[common_ids]
    proteome = proteome.loc[common_ids]

    train_mask = meta["split_final"].astype(str).eq("train")
    missing_rate = proteome.loc[train_mask].isna().mean(axis=0)
    protein_names = missing_rate.index[missing_rate < args.missing_threshold]
    y_log2 = np.log2(proteome.loc[:, protein_names].astype(np.float32))
    mask = y_log2.notna()

    fc_started = time.perf_counter()
    pairs = build_matched_control_pairs(meta, train_mask)
    result = compute_fold_change(
        meta, y_log2, mask, train_mask=train_mask, pairs=pairs
    )
    print(f"构造训练集FC标签: {time.perf_counter() - fc_started:.1f}s", flush=True)

    perturbation = meta["perturbation_no_concentration"].astype(str).str.lower()
    train_treatments = train_mask & ~perturbation.isin(["water", "dmso"])
    train_treatments &= ~perturbation.str.contains("quality|qc", regex=True, na=False)

    matched = len(pairs)
    total_treatments = int(train_treatments.sum())
    valid_fc = int(result["fc_mask"].to_numpy().sum())
    possible_fc = int(np.prod(result["fc_mask"].shape))
    control_counts = pairs["n_controls"].value_counts().sort_index()

    assert set(result["fc_true"].index).issubset(set(meta.index[train_mask]))
    assert result["fc_true"].shape == result["fc_mask"].shape
    fc_array = result["fc_true"].to_numpy()
    fc_mask_array = result["fc_mask"].to_numpy()
    assert np.isfinite(fc_array[fc_mask_array]).all()

    print(f"训练样本: {int(train_mask.sum())}")
    print(f"按训练集缺失率保留蛋白: {len(protein_names)}")
    print(f"训练处理样本: {total_treatments}")
    print(f"精确匹配处理样本: {matched} ({matched / total_treatments:.2%})")
    print("每个处理对应的对照重复数: " + ", ".join(
        f"{int(n_controls)}个对照={int(count)}个处理"
        for n_controls, count in control_counts.items()
    ))
    print(f"有效FC蛋白-样本对: {valid_fc}/{possible_fc} ({valid_fc / possible_fc:.2%})")
    print("数据隔离检查: FC标签全部来自train，未读取test真值 [OK]")


if __name__ == "__main__":
    main()
