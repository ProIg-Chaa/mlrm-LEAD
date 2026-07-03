from pathlib import Path
import pyarrow.parquet as pq

root = Path("/share/home/wangzixu/liudinghao/gushuo/datasets/sources/lmms-lab__POPE")
for path in sorted(root.glob("**/*.parquet")):
    print(f"\nFILE {path}")
    table = pq.read_table(path)
    print(table.schema)
    print("rows", table.num_rows)
    row = table.slice(0, 1).to_pylist()[0]
    for key, value in row.items():
        text = repr(value)
        if len(text) > 500:
            text = text[:500] + "..."
        print(f"{key}: {type(value).__name__} {text}")
