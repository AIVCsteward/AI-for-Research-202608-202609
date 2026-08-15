"""
向后兼容模块：所有功能已迁移到 baseline/data.py

本模块保留以下用途:
  1. 作为 import 别名，使 `from baseline.preprocess import ...` 继续有效
  2. `__main__` 块可用于临时数据探索: `python -m baseline.preprocess`

  预处理的唯一权威实现位于 baseline/data.py
"""
from baseline.data import (        # noqa: F401 — re-export
    load_raw_data as load_data,
    preprocess as align_and_filter,
    get_split_masks,
    identify_controls,
    identify_treatments,
    DATA_DIR,
)

if __name__ == "__main__":
    # 临时数据探索
    meta, prot = load_data()
    y_log2, mask, meta, proteins, train_mask = align_and_filter(meta, prot)
    splits = get_split_masks(meta)

    print(f"\n各 split 样本数:")
    for k, v in splits.items():
        print(f"  {k}: {v.sum()}")

    print(f"\n对照组样本数: {identify_controls(meta).sum()}")
    print(f"处理组样本数: {identify_treatments(meta).sum()}")
