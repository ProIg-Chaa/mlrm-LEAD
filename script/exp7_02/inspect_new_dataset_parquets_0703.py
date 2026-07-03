import pyarrow.parquet as pq

paths = [
    "/share/home/wangzixu/liudinghao/gushuo/datasets/sources/FanqingM__MMK12/data/test-00000-of-00001.parquet",
    "/share/home/wangzixu/liudinghao/gushuo/datasets/sources/suyc21__VMCBench/data/dev-00000-of-00001.parquet",
    "/share/home/wangzixu/liudinghao/gushuo/datasets/sources/suyc21__VMCBench/data/test-00000-of-00002.parquet",
]

for path in paths:
    print("===", path)
    table = pq.read_table(path)
    print(table.schema)
    print("rows", table.num_rows)
    row = table.slice(0, 1).to_pylist()[0]
    for key, value in row.items():
        if isinstance(value, (bytes, bytearray)):
            print(key, "<bytes>", len(value))
        else:
            text = str(value).replace("\n", " ")
            print(key, text[:600])
