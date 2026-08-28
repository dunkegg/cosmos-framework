import json
import numpy as np

a = np.array(json.load(open("datasets/action/av_480_20260501_7.json")))

print(a[:10, :3])

d = np.diff(a[:, :3], axis=0)

print("diff first 10:")
print(d[:10])

print("diff norm:")
print(np.linalg.norm(d, axis=1)[:20])