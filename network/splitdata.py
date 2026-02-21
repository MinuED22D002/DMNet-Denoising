import re
import random
from pathlib import Path

SEED = 42
TRAIN, VAL, TEST = 0.70, 0.15, 0.15
INPUT_FILE = Path("allfolders.txt")

# Read lines
lines = [line.strip() for line in INPUT_FILE.read_text().splitlines() if line.strip()]

# Group key = everything before "_gaussian_"
def group_key(name):
    return re.split(r"_gaussian_", name)[0]

groups = {}
for name in lines:
    groups.setdefault(group_key(name), []).append(name)

group_ids = list(groups.keys())

random.seed(SEED)
random.shuffle(group_ids)

n = len(group_ids)
n_train = int(round(TRAIN * n))
n_val   = int(round(VAL * n))
n_test  = n - n_train - n_val

train_ids = set(group_ids[:n_train])
val_ids   = set(group_ids[n_train:n_train+n_val])
test_ids  = set(group_ids[n_train+n_val:])

train, val, test = [], [], []

for gid, names in groups.items():
    if gid in train_ids:
        train.extend(names)
    elif gid in val_ids:
        val.extend(names)
    else:
        test.extend(names)

Path("train.txt").write_text("\n".join(sorted(train)) + "\n")
Path("val.txt").write_text("\n".join(sorted(val)) + "\n")
Path("test.txt").write_text("\n".join(sorted(test)) + "\n")

print("Total entries:", len(lines))
print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))
