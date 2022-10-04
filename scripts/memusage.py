import pandas as pd

# get sysargs
import sys

path = f"../data/{sys.argv[1]}/memusage.csv"

file = pd.read_csv(path)
file = file.sort_values(by=["max_memory_mb"], ascending=False)
print(file)
pd.set_option("display.max_colwidth", None)
print(file["task_key"])
