"""Additional statistics for version_4 (computed from the stored version_3 results; no re-runs)."""
import json, sys, os
import numpy as np, pandas as pd
from scipy import stats
R = sys.argv[1]; D = sys.argv[2]
S = ["electricity", "weather", "airlines", "covertype"]
BASE = ["Incremental XGBoost", "ADWIN-reset XGBoost", "Sliding XGBoost", "Online MLP", "HAT", "ARF"]
out = {}
for s in S:
    M = json.load(open(f"{R}/main_{s}.json"))["methods"]
    o = out[s] = {}
    # mean +- std for all primary metrics
    o["table"] = {k: {m: (100*np.mean([r[m] for r in v["runs"]]) if m != "kappa" else np.mean([r[m] for r in v["runs"]]),
                          100*np.std([r[m] for r in v["runs"]], ddof=1) if m != "kappa" else np.std([r[m] for r in v["runs"]], ddof=1))
                      for m in ("acc", "f1", "kappa")} for k, v in M.items()}
    best = max(BASE, key=lambda k: M[k]["mean"]["acc"]); o["best"] = best
    a = np.array([r["acc"] for r in M["AX-NXGB"]["runs"]]); b = np.array([r["acc"] for r in M[best]["runs"]])
    d = 100*(a-b); n = len(d); h = stats.t.ppf(0.975, n-1)*d.std(ddof=1)/np.sqrt(n)
    o["diff"] = (d.mean(), d.mean()-h, d.mean()+h, stats.ttest_rel(a, b).pvalue)
    ca = np.array(M["AX-NXGB"]["runs"][0]["acc_chunks"]); cb = np.array(M[best]["runs"][0]["acc_chunks"])
    o["chunk_wilcoxon_p"] = stats.wilcoxon(ca, cb).pvalue; o["chunk_win"] = float((ca > cb).mean()); o["n_chunks"] = len(ca)
    o["chunk_tie"] = float((ca == cb).mean())
    # cost: ms per chunk (learn+predict), refits
    nch = len(ca)+1
    o["cost"] = {k: (1e3*(v["mean"]["t_learn"]+v["mean"]["t_pred"])/nch, v["mean"].get("n_refit")) for k, v in M.items()}
    # post-alarm chunks (alarms of AX-NXGB seed 0 from the explanation run, same config/seed)
    E = json.load(open(f"{R}/explain_{s}.json"))
    al = np.where(E["log"]["alarm"])[0]
    o["n_alarm_seed0"] = (len(al), M["AX-NXGB"]["runs"][0]["n_alarm"])
    idx = sorted({c+k for c in al for k in range(1, 5) if 1 <= c+k < nch})   # chunks c+1..c+4
    o["post_alarm"] = {k: 100*np.mean(np.array(M[k]["runs"][0]["acc_chunks"])[np.array(idx)-1]) for k in ["AX-NXGB", "AX-NXGB-Lite", best, "ARF", "Static XGBoost"]} if idx else None
    o["post_alarm_nchunks"] = len(idx)
    # explanation uncertainty: mean +- std over explained chunks, 95% CI of mean
    L = E["log"]
    o["cos"] = {k: (np.mean(L["cos_"+k]), np.std(L["cos_"+k], ddof=1), len(L["cos_"+k])) for k in ("frozen", "static", "alarm", "dyn")}
    o["dyn_vs_static_wilcoxon"] = stats.wilcoxon(L["cos_dyn"], L["cos_static"]).pvalue
    if L["exact_cos"]:
        t = np.array(L["t_exact"])
        o["exact"] = dict(mean=np.mean(L["exact_cos"]), sd=np.std(L["exact_cos"], ddof=1), min=np.min(L["exact_cos"]), n=len(L["exact_cos"]),
                          spear=np.mean(L["exact_spear"]), speedup=float(np.mean(t[:, 1]/t[:, 0])), t_fast=1e3*t[:, 0].mean(), t_exact=1e3*t[:, 1].mean())
    o["eff_resid"] = (np.mean(L["eff_resid"]), np.max(L["eff_resid"]))
    o["t_fast_ms"] = 1e3*np.mean(L["t_fast"])
    I = np.array(L["I_dyn"]); o["zero_profiles"] = int((np.linalg.norm(I, axis=1) < 1e-12).sum())
    o["edi"] = E["summary"]["edi_alarm_auc"], E["summary"]["edi_at_alarm"], E["summary"]["edi_quiet"]
# data checks + label-only reference classifiers
F = dict(electricity="electricity.csv", weather="weather.csv", airlines="airlines.csv", covertype="covtype.csv")
for s in S:
    df = pd.read_csv(os.path.join(D, F[s])); o = out[s]
    if s == "electricity": y = (df["class"].str.upper() == "UP").astype(int).values
    elif s == "weather": y = df["target"].values
    elif s == "airlines": y = df["Delay"].values[:100000]
    else: y = (df.iloc[:100000, -1].values == 2).astype(int)
    o["n"], o["nan"], o["pos_rate"] = len(y), int(df.isna().sum().sum()), float(y.mean())
    yt = y[250:]
    o["nochange_instance"] = 100*np.mean(y[250:] == y[249:-1])           # needs y_{i-1}: NOT admissible under chunk labels
    prev_last = np.array([y[(i//250)*250-1] for i in range(250, len(y))])
    o["nochange_chunk"] = 100*np.mean(yt == prev_last)
    maj = np.array([int(y[(i//250-1)*250:(i//250)*250].mean() >= 0.5) for i in range(250, len(y))])
    o["majority_prev_chunk"] = 100*np.mean(yt == maj)
json.dump(out, open("stats_v4.json", "w"), indent=1, default=float)
for s in S:
    o = out[s]
    print(f"\n=== {s}: n={o['n']} NaN={o['nan']} pos={o['pos_rate']:.3f} best={o['best']}")
    print(" diff vs best: %.2f [%.2f, %.2f] p=%.2g | chunk Wilcoxon p=%.2g win=%.2f tie=%.2f nch=%d" % (*o["diff"], o["chunk_wilcoxon_p"], o["chunk_win"], o["chunk_tie"], o["n_chunks"]))
    print(" no-change(inst)=%.2f no-change(chunk)=%.2f majority(prev chunk)=%.2f" % (o["nochange_instance"], o["nochange_chunk"], o["majority_prev_chunk"]))
    print(" alarms seed0", o["n_alarm_seed0"], "post-alarm chunks", o["post_alarm_nchunks"], {k: round(v, 2) for k, v in (o["post_alarm"] or {}).items()})
    print(" cost ms/chunk,refits:", {k: (round(v[0], 1), v[1]) for k, v in o["cost"].items()})
    print(" cos:", {k: tuple(round(x, 3) for x in v[:2]) for k, v in o["cos"].items()}, "p(dyn vs static)=%.2g" % o["dyn_vs_static_wilcoxon"])
    print(" exact:", o.get("exact"), "eff", o["eff_resid"], "tfast", o["t_fast_ms"], "zeroI", o["zero_profiles"], "edi", o["edi"])
    for k, v in o["table"].items():
        print("   %-22s" % k, " ".join("%s=%.2f±%.2f" % (m, *v[m]) if m != "kappa" else "k=%.3f±%.3f" % v[m] for m in v))
