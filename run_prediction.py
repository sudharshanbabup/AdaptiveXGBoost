"""
Prediction experiments (main comparison, ablation, DCL generality, refit-cost trade-off).
Usage:  python run_prediction.py <dataset> <main|ablation|dcl|tradeoff>
All results are written to results/<experiment>_<dataset>.json
"""
import sys, json, time, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
from data import load_stream
from models import (AdaptiveNeuralXGB, StaticXGB, WindowXGB, AdwinResetXGB, IncrementalXGB,
                    OnlineMLP, RiverWrap, DCLWrap)
from stream_eval import prequential, summarize

CHUNK, HORIZON = 250, 6000
MAX_N = {"electricity": None, "weather": None, "airlines": 100000, "covertype": 100000}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def experiments(p, cont):
    P = lambda **k: (lambda s: AdaptiveNeuralXGB(p, seed=s, horizon=HORIZON, dcl_cols=cont, **k))
    nd = p + 3 * len(cont)
    main = {
        "Static XGBoost": lambda s: StaticXGB(p, s, n0=HORIZON),
        "Incremental XGBoost": lambda s: IncrementalXGB(p, s),
        "ADWIN-reset XGBoost": lambda s: AdwinResetXGB(p, s, w_max=HORIZON),
        "Sliding XGBoost": lambda s: WindowXGB(p, s, window=HORIZON),
        "Online MLP": lambda s: OnlineMLP(p, s),
        "HAT": lambda s: RiverWrap(p, s, "hat"),
        "ARF": lambda s: RiverWrap(p, s, "arf"),
        "AX-NXGB-Lite": P(refit_every="adaptive"),
        "AX-NXGB": P(refit_every=1),
    }
    ablation = {
        "Full (AX-NXGB)": P(refit_every=1),
        "w/o DCL": P(refit_every=1, use_dcl=False),
        "w/o neural branch": P(refit_every=1, use_neural=False),
        "w/o drift gate": P(refit_every=1, gate=False),
        "latent fed to trees": P(refit_every=1, tree_latent=True),
        "trees on latent only (v2)": P(refit_every=1, latent_only=True),
        "Lite": P(refit_every="adaptive"),
        "Lite w/o neural": P(refit_every="adaptive", use_neural=False),
        "Lite w/o ADWIN trigger": P(refit_every=8, refit_on_drift=False),
        "frozen trees": P(refit_every=10 ** 9, refit_on_drift=False),
        "frozen trees w/o neural": P(refit_every=10 ** 9, refit_on_drift=False, use_neural=False),
        "frozen trees w/o gate": P(refit_every=10 ** 9, refit_on_drift=False, gate=False),
    }
    dcl = {
        "Sliding XGBoost": lambda s: WindowXGB(p, s, window=HORIZON),
        "Sliding XGBoost + DCL": lambda s: DCLWrap(WindowXGB(nd, s, window=HORIZON), p, cols=cont),
        "Online MLP": lambda s: OnlineMLP(p, s),
        "Online MLP + DCL": lambda s: DCLWrap(OnlineMLP(nd, s), p, cols=cont),
        "ARF": lambda s: RiverWrap(p, s, "arf"),
        "ARF + DCL": lambda s: DCLWrap(RiverWrap(nd, s, "arf"), p, cols=cont),
    }
    tradeoff = {}
    for R in (1, 2, 4, 8, 16, 32, 64):
        tradeoff[f"hybrid|R={R}"] = P(refit_every=R, refit_on_drift=False)
        tradeoff[f"tree|R={R}"] = P(refit_every=R, refit_on_drift=False, use_neural=False)
    tradeoff["hybrid|adaptive"] = P(refit_every="adaptive")
    tradeoff["tree|adaptive"] = P(refit_every="adaptive", use_neural=False)
    return dict(main=(main, 5), ablation=(ablation, 3), dcl=(dcl, 3), tradeoff=(tradeoff, 3))


def run(dataset, exp):
    X, y, names, cont = load_stream(dataset, MAX_N[dataset])
    methods, n_seeds = experiments(X.shape[1], cont)[exp]
    path = os.path.join(OUT, f"{exp}_{dataset}.json")
    res = dict(dataset=dataset, n=int(len(y)), p=int(X.shape[1]), chunk=CHUNK, horizon=HORIZON, methods={})
    if os.path.exists(path):                      # resume: keep finished methods
        res = json.load(open(path))
    res["methods"] = {k: v for k, v in res["methods"].items() if k in methods}
    for name, make in methods.items():
        if name in res["methods"]:
            continue
        rows = []
        for seed in range(n_seeds):
            t0 = time.time()
            m = make(seed)
            r = prequential(m, X, y, CHUNK)
            s = summarize(r["P"], y, r["start"], CHUNK)
            row = dict(seed=seed, acc=s["acc"], f1=s["f1"], kappa=s["kappa"], auc=s["auc"], worst10=s["worst10"],
                       t_pred=r["t_pred"], t_learn=r["t_learn"], wall=time.time() - t0)
            core = m.base if isinstance(m, DCLWrap) else m
            if isinstance(core, AdaptiveNeuralXGB):
                row.update(n_refit=core.n_refit, n_alarm=len(core.drift_log),
                           alpha=float(np.mean(core.alpha_log)), gamma=float(np.mean(core.gamma_log)))
            if seed == 0:
                row["acc_chunks"] = [float(v) for v in s["acc_chunks"]]
            rows.append(row)
            print(f"[{exp}/{dataset}] {name:28s} seed={seed} acc={s['acc']:.4f} f1={s['f1']:.4f} "
                  f"kappa={s['kappa']:.4f} auc={s['auc']:.4f} t={row['wall']:.0f}s", flush=True)
        agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
               for k in rows[0] if k not in ("seed", "acc_chunks")}
        res["methods"][name] = dict(runs=rows, mean={k: v[0] for k, v in agg.items()}, std={k: v[1] for k, v in agg.items()})
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
