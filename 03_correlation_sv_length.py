import pickle
import numpy as np
import pandas as pd
from scipy import stats


OUTPUT_ROOT = r"D:\PycharmProjects\xAI-PPM\outputs"
with open(f"{OUTPUT_ROOT}/shap_values/bpi17_transition/segment_sv_results.pkl", "rb") as f:
    payload = pickle.load(f)

# Build the flat table
rows = []
for name in ["tp", "tn", "fp", "fn"]:
    for ci, res in enumerate(payload["sv_results"][name]):
        for si, (sv, seg) in enumerate(zip(
            np.asarray(res["segment_sv"]).ravel(), res["segment_ids"]
        )):
            rows.append({"sample": name, "case": ci, "seg_length": len(seg), "abs_sv": abs(sv)})

df = pd.DataFrame(rows)

# Quick look
print(df.groupby("sample")[["seg_length", "abs_sv"]].describe())
print(stats.spearmanr(df["seg_length"], df["abs_sv"]))