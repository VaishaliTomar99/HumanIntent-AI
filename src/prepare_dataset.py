import os, csv, random
from pathlib import Path

base = Path("data/A-Dataset-for-Automatic-Violence-Detection-in-Videos/violence-detection-dataset")

# violent and non-violent video lists
violent = list((base/"violent"/"cam1").glob("*.mp4")) + list((base/"violent"/"cam2").glob("*.mp4"))
nonviolent = list((base/"non-violent"/"cam1").glob("*.mp4")) + list((base/"non-violent"/"cam2").glob("*.mp4"))

all_data = [(str(v), 1) for v in violent] + [(str(n), 0) for n in nonviolent]
random.shuffle(all_data)

# split 70/15/15
n = len(all_data)
train = all_data[:int(0.7*n)]
val   = all_data[int(0.7*n):int(0.85*n)]
test  = all_data[int(0.85*n):]

def save_csv(path, data):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename","label"])
        writer.writerows(data)

Path("data").mkdir(exist_ok=True)
save_csv("data/train.csv", train)
save_csv("data/val.csv", val)
save_csv("data/test.csv", test)

print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
