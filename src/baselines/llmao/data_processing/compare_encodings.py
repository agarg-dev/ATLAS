import csv

for fname in ["data/beetlebox/code_bugline_small_250.csv", "data/defects4j/data_defects4j.csv"]:
    print(f"\nInspecting {fname}:")
    with open(fname, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            print(f"  row {i+1}: {len(row)} columns → {row[:3]!r}")
            if i >= 5:
                break
