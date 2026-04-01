import pandas as pd, pathlib, textwrap, sys, json, csv

csv_path = pathlib.Path(r"C:\atlas\LLMAO\data\defects4j\data_defects4j.csv")
df = pd.read_csv(csv_path, nrows=5)   # just peek at the first 5 rows
print(df.columns)                     # what columns really exist?
print(df.head(3))
