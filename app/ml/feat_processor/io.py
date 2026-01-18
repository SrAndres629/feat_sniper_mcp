import os
import pandas as pd
import json
import gzip
from typing import List, Dict

def export_parquet(df: pd.DataFrame, path: str):
    try:
        df.to_parquet(path, compression="snappy")
        print(f"✅ Exported Parquet: {path}")
    except Exception as e:
        print(f"❌ Failed export Parquet: {e}")

def export_jsonl_gz(data: List[Dict], path: str):
    try:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for rec in data: f.write(json.dumps(rec) + "\n")
        print(f"✅ Exported JSONL.GZ: {path}")
    except Exception as e:
        print(f"❌ Failed export JSONL: {e}")
