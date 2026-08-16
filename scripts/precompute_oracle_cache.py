"""Generate the read-only oracle cache.  Run once before the four plan scripts.

    python scripts/precompute_oracle_cache.py
"""
from scripts.oracle_common import build_cache, cache_exists

if __name__ == "__main__":
    if cache_exists():
        print("缓存已存在，跳过生成。如需重建请删除 data/external/oracle_cache/")
    else:
        build_cache()
