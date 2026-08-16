"""一键串行跑完四个 SNR 实验（预热缓存 + 四 plan + 汇总）。

    python -m scripts.run_all_oracle_plans

四个 plan 都只读缓存、写独立输出，单窗口串行即可，无需多窗口。
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import plan_a_snr_filter, plan_b_consensus, plan_c_batch_correction, plan_d_lowrank
from scripts.oracle_common import build_cache, cache_exists

OUT_DIR = Path("experiments/outputs/oracle")


def main():
    if not cache_exists():
        print("缓存不存在，先生成……")
        build_cache()
    else:
        print("缓存已存在，跳过生成")

    print("\n========== Plan A：蛋白级信噪比筛选 ==========")
    plan_a_snr_filter.main()
    print("\n========== Plan B：跨重复共识去噪 ==========")
    plan_b_consensus.main()
    print("\n========== Plan C：批次效应校正 ==========")
    plan_c_batch_correction.main()
    print("\n========== Plan D：低秩分解 ==========")
    plan_d_lowrank.main()

    print("\n========== 汇总 ==========")
    for name, key in [
        ("Plan A 信噪比", "K=500"),
        ("Plan B 共识去噪", "denoised_leave_one_out_median"),
        ("Plan C 批次校正", "batch_corrected_oracle"),
        ("Plan D 低秩分解", "k=50"),
    ]:
        f = OUT_DIR / {
            "Plan A 信噪比": "plan_a_snr.json",
            "Plan B 共识去噪": "plan_b_consensus.json",
            "Plan C 批次校正": "plan_c_batch.json",
            "Plan D 低秩分解": "plan_d_lowrank.json",
        }[name]
        if not f.exists():
            print(f"{name}: 缺失")
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        base = data.get("baseline_oracle", {})
        res = data.get(key)
        print(f"{name}: baseline={base.get('mean')} → 处理后={res.get('mean') if res else None}")


if __name__ == "__main__":
    main()
