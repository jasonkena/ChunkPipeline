import pandas as pd

path = "../data/seg_den/memusage.csv"

file = pd.read_csv(path)
file = file.sort_values(by=["max_memory_mb"], ascending=False)
print(file)
