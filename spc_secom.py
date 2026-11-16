"""SPC drift alerts on REAL semiconductor process data (UCI SECOM, 1567 runs x 590 sensors).

EWMA control chart (lambda=0.2, L=3) with limits from a 300-run baseline, next to the individuals 3-sigma limits.
The sensor shown is the one most correlated with the real pass/fail label (not cherry-picked for the alert timing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = 300
X = pd.read_csv("data/secom.data", sep=" ", header=None)
lab = pd.read_csv("data/secom_labels.data", sep=" ", header=None, names=["y", "t"])
lab["t"] = pd.to_datetime(lab["t"].str.strip('"'), format="%d/%m/%Y %H:%M:%S")
fail = (lab["y"] == 1).to_numpy()


def first_run(z: np.ndarray, n: int) -> int | None:
    s = np.sign(z)
    for i in range(n - 1, len(z)):
        w = s[i - n + 1 : i + 1]
        if abs(w.sum()) == n:
            return i
    return None


def first_trend(v: np.ndarray, n: int = 6) -> int | None:
    d = np.sign(np.diff(v))
    for i in range(n - 2, len(d)):
        w = d[i - n + 2 : i + 1]
        if abs(w.sum()) == n - 1:
            return i + 1
    return None


def analyse(col: pd.Series):
    v = col.to_numpy(float)
    ok = ~np.isnan(v)
    if ok.mean() < 0.95:
        return None
    v = np.where(ok, v, np.nanmedian(v))
    mu, sd = v[:BASE].mean(), v[:BASE].std(ddof=1)
    if sd < 1e-9:
        return None
    z = (v - mu) / sd
    out = np.where(np.abs(z[BASE:]) > 3)[0]
    r1 = int(out[0]) + BASE if len(out) else None
    r2 = first_run(z[BASE:], 8)
    r2 = r2 + BASE if r2 is not None else None
    return dict(v=v, mu=mu, sd=sd, z=z, r1=r1, r2=r2)



cands = []
for j in X.columns:
    a = analyse(X[j])
    if a is None:
        continue
    cands.append((abs(np.corrcoef(a["v"], fail)[0, 1]), j, a))
cands.sort(key=lambda c: -c[0])
corr, j, a = cands[0]
z = a["z"]; lam = 0.2
ew = np.zeros_like(z); ew[0] = lam * z[0]
for i in range(1, len(z)):
    ew[i] = lam * z[i] + (1 - lam) * ew[i - 1]
sig = np.sqrt(lam / (2 - lam))          # steady-state EWMA sigma in z units
alarm = np.where(np.abs(ew[BASE:]) > 3 * sig)[0]
ewma_alarm = int(alarm[0]) + BASE if len(alarm) else None
t = lab["t"]
print(f"sensor {j} corr(fail)={corr:.3f}; EWMA first alarm run={ewma_alarm}; individual 3-sigma first breach run={a['r1']}; "
      f"fraction of post-baseline EWMA points out of limits={np.mean(np.abs(ew[BASE:]) > 3*sig):.2f}")

fig, axs = plt.subplots(2, 1, figsize=(10, 5.6), dpi=200, sharex=True)
axs[0].plot(t, a["v"], lw=0.6, color="#9db7e8", marker="o", ms=1.5)
axs[0].axhline(a["mu"], color="#555", lw=1)
axs[0].axhline(a["mu"] + 3 * a["sd"], color="#c0392b", ls="--", lw=1); axs[0].axhline(a["mu"] - 3 * a["sd"], color="#c0392b", ls="--", lw=1)
axs[0].set_title(f"Individuals chart, SECOM sensor #{j} (real fab data)", fontsize=9)
axs[1].plot(t, a["mu"] + a["sd"] * ew, color="#1f5fd1", lw=1.2, label="EWMA (λ=0.2)")
axs[1].axhline(a["mu"], color="#555", lw=1, label="CL")
axs[1].axhline(a["mu"] + 3 * sig * a["sd"], color="#c0392b", ls="--", lw=1, label="UCL / LCL")
axs[1].axhline(a["mu"] - 3 * sig * a["sd"], color="#c0392b", ls="--", lw=1)
for ax in axs:
    ax.axvspan(t.iloc[0], t.iloc[BASE - 1], color="#d9e6ff", alpha=0.5)
if ewma_alarm is not None:
    axs[1].axvline(t.iloc[ewma_alarm], color="#e08a2c", lw=1.8, label=f"first EWMA alarm – run {ewma_alarm}")
axs[1].set_title("EWMA chart: drift / small shifts (shaded = baseline)", fontsize=9)
axs[1].legend(fontsize=7, frameon=False, ncol=4, loc="upper left")
for ax in axs: ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/spc_control_chart.png")

# yield view: p-chart of weekly fail rate
w = pd.DataFrame({"t": t, "f": fail}).set_index("t").resample("W")["f"].agg(["sum", "count"])
w = w[w["count"] >= 20]
p0 = fail[:BASE].mean(); n = w["count"]
ucl = p0 + 3 * np.sqrt(p0 * (1 - p0) / n)
fig, ax = plt.subplots(figsize=(10, 3.4), dpi=200)
rate = w["sum"] / n
ax.plot(w.index, rate, marker="o", color="#1f5fd1", label="weekly fail rate")
ax.plot(w.index, ucl, color="#c0392b", ls="--", label="UCL (p-chart, baseline p̄)")
ax.axhline(p0, color="#555", label="p̄ baseline")
ax.set_title("Lot-level yield p-chart, real SECOM pass/fail labels", fontsize=10)
ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/spc_weekly_fail_rate.png")
pd.Series(dict(sensor=int(j), corr_with_fail=float(corr), ewma_first_alarm_run=ewma_alarm,
               individual_first_breach_run=a["r1"], runs=len(X), fail_rate=float(fail.mean()))).to_json("figures/spc_summary.json")
