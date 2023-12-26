import json
import os

l = []
i = True
idx = 0
for dirpath, dirnames, filenames in os.walk("data/train"):
    if i:
        i = False
        continue
    for filename in filenames:
        l.append([os.path.join(dirpath, filename).replace("\\", "/"), idx])
    idx += 1
res = json.dumps(l)
with open("data/train_list.txt", "w") as f:
    f.write(res)

l = []
i = True
idx = 0
for dirpath, dirnames, filenames in os.walk("data/test"):
    if i:
        i = False
        continue
    for filename in filenames:
        l.append([os.path.join(dirpath, filename).replace("\\", "/"), idx])
    idx += 1
res = json.dumps(l)
with open("data/test_list.txt", "w") as f:
    f.write(res)