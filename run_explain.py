"""
Explainability experiments: Dynamic SHAP vs. stale reference sets, agreement with exact
end-to-end Shapley values, and the Explanation Drift Index (EDI).
Usage:  python run_explain.py <dataset> [nodcl]   ->  results/explain_<dataset>.json
"""
import sys, json, os, time, copy, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from data import load_stream
from models import AdaptiveNeuralXGB
from stream_eval import prequential, summarize
from explain import (feature_groups, hybrid_shap, exact_group_shapley, importance_profile, edi, cosine_rows)
from run_prediction import CHUNK, HORIZON, MAX_N, OUT

N_EXPLAIN, N_REF, REF_MEM = 100, 100, 500     # explained rows / reference rows / reference memory
START_CHUNK, EXACT_EVERY, N_EXACT = 8, 10, 40


def sub(A, n, rng):
    return A if len(A) <= n else A[rng.choice(len(A), n, replace=False)]


def run(dataset, seed=0, use_dcl=True):
    X, y, names, cont = load_stream(dataset, MAX_N[dataset])
    p = X.shape[1]
    model = AdaptiveNeuralXGB(p, seed=seed, horizon=HORIZON, dcl_cols=cont, refit_every=1, use_dcl=use_dcl)
    G = feature_groups(p, model.dcl)
    rng = np.random.default_rng(seed)
    st = dict(B_static=None, B_alarm=None, pool=np.empty((0, model.pa), np.float32), prevI=None, frozen=None)
    log = dict(chunk=[], acc=[], alarm=[], alpha=[], gamma=[], edi=[], edi_feat=[], I_dyn=[], I_static=[], I_oracle=[],
               gap_frozen=[], gap_static=[], gap_alarm=[], gap_dyn=[], cos_frozen=[], cos_static=[], cos_alarm=[],
               cos_dyn=[], top1_frozen=[], top1_static=[], top1_alarm=[], top1_dyn=[], eff_resid=[], t_fast=[],
               exact_chunk=[], exact_cos=[], exact_spear=[], exact_top1=[], t_exact=[])
    do_exact = p <= 8

    def hook(c, s, e, m, Xc, yc, prob, phase):
        A = m.last["A"]
        if phase == "post":                       # ---- reference-set maintenance (after labels)
            if st["B_static"] is None and len(st["pool"]) + len(A) >= REF_MEM:
                st["B_static"] = sub(np.vstack([st["pool"], A]), N_REF, rng)       # frozen forever
                st["B_alarm"] = st["B_static"]
                st["frozen"] = copy.deepcopy(m)                                   # non-adaptive explainer
            if m.drift_flag:                      # ADWIN alarm: discard pre-drift references
                st["pool"] = A.copy()
                st["B_alarm"] = sub(A, N_REF, rng)                                 # v2 rule
            else:
                st["pool"] = np.vstack([st["pool"], A])[-REF_MEM:]
            log["alarm"].append(bool(m.drift_flag))
            return
        if c < START_CHUNK or m.clf is None:      # ---- explanation of the incoming chunk
            return
        Ae = sub(A, N_EXPLAIN, rng)
        refs = dict(static=st["B_static"], alarm=st["B_alarm"], dyn=sub(st["pool"], N_REF, rng),
                    oracle=sub(A, N_REF, rng))
        Fbar = float(m.margin(A).mean())
        out = {}
        for k, B in refs.items():
            t0 = time.perf_counter()
            Phi, base = hybrid_shap(m, Ae, B, G)
            if k == "dyn":
                log["t_fast"].append(time.perf_counter() - t0)
                log["eff_resid"].append(float(np.abs(Phi.sum(1) + base - m.margin(Ae)).mean()))
            out[k] = (Phi, base)
        out["frozen"] = hybrid_shap(st["frozen"], Ae, st["B_static"], G)
        Po = out["oracle"][0]
        for k in ("frozen", "static", "alarm", "dyn"):
            Phi, base = out[k]
            log["gap_" + k].append(abs(base - Fbar))
            log["cos_" + k].append(float(cosine_rows(Phi, Po).mean()))
            log["top1_" + k].append(float((np.abs(Phi).argmax(1) == np.abs(Po).argmax(1)).mean()))
        I = importance_profile(out["dyn"][0])
        if st["prevI"] is not None:
            v, d = edi(st["prevI"], I)
        else:
            v, d = 0.0, np.zeros(p)
        st["prevI"] = I
        log["chunk"].append(c); log["acc"].append(float(((prob >= 0.5) == yc).mean()))
        log["alpha"].append(m.alpha); log["gamma"].append(m.gamma)
        log["edi"].append(v); log["edi_feat"].append(d.tolist())
        log["I_dyn"].append(I.tolist()); log["I_static"].append(importance_profile(out["static"][0]).tolist())
        log["I_oracle"].append(importance_profile(Po).tolist())
        if do_exact and c % EXACT_EVERY == 0:     # ---- agreement with exact end-to-end Shapley
            Ax, Bx = Ae[:N_EXACT], refs["dyn"][:N_EXACT]
            t0 = time.perf_counter(); Pf, _ = hybrid_shap(m, Ax, Bx, G); tf = time.perf_counter() - t0
            t0 = time.perf_counter(); Pe = exact_group_shapley(m, Ax, Bx, G); te = time.perf_counter() - t0
            log["exact_chunk"].append(c)
            log["exact_cos"].append(float(cosine_rows(Pf, Pe).mean()))
            log["exact_spear"].append(float(np.nanmean([spearmanr(a, b)[0] for a, b in zip(Pf, Pe)])))
            log["exact_top1"].append(float((np.abs(Pf).argmax(1) == np.abs(Pe).argmax(1)).mean()))
            log["t_exact"].append([tf, te])
        if c % 20 == 0:
            print(f"[explain/{dataset}] chunk {c}: acc={log['acc'][-1]:.3f} EDI={v:.4f} cos(static/alarm/dyn)="
                  f"{log['cos_static'][-1]:.3f}/{log['cos_alarm'][-1]:.3f}/{log['cos_dyn'][-1]:.3f}", flush=True)

    r = prequential(model, X, y, CHUNK, hook=hook)
    s = summarize(r["P"], y, r["start"], CHUNK)
    ch = np.array(log["chunk"]); al = np.array(log["alarm"])[ch]; E = np.array(log["edi"]); err = 1 - np.array(log["acc"])
    near = np.array([al[max(0, i - 1):i + 2].any() for i in range(len(al))])
    summ = dict(acc=s["acc"], n_alarm=int(np.sum(log["alarm"])), n_explained_chunks=len(ch))
    for k in ("frozen", "static", "alarm", "dyn"):
        summ["gap_" + k] = float(np.mean(log["gap_" + k])); summ["cos_" + k] = float(np.mean(log["cos_" + k]))
        summ["top1_" + k] = float(np.mean(log["top1_" + k]))
        summ["cos_p10_" + k] = float(np.percentile(log["cos_" + k], 10))
    summ["eff_resid"] = float(np.mean(log["eff_resid"])); summ["t_fast_ms"] = 1e3 * float(np.mean(log["t_fast"]))
    summ["edi_mean"] = float(E[1:].mean()); summ["edi_median"] = float(np.median(E[1:]))
    summ["edi_err_spearman"] = float(spearmanr(E[1:], err[1:])[0])
    summ["edi_alarm_auc"] = float(roc_auc_score(near[1:], E[1:])) if 0 < near[1:].sum() < len(near) - 1 else None
    summ["edi_at_alarm"] = float(E[1:][near[1:]].mean()) if near[1:].any() else None
    summ["edi_quiet"] = float(E[1:][~near[1:]].mean())
    if log["exact_cos"]:
        tt = np.array(log["t_exact"])
        summ.update(exact_cos=float(np.mean(log["exact_cos"])), exact_spear=float(np.mean(log["exact_spear"])),
                    exact_top1=float(np.mean(log["exact_top1"])), exact_cos_min=float(np.min(log["exact_cos"])),
                    t_fast_exactset_ms=1e3 * float(tt[:, 0].mean()), t_exact_ms=1e3 * float(tt[:, 1].mean()))
    print(json.dumps(summ, indent=1))
    json.dump(dict(dataset=dataset, features=names, summary=summ, log=log),
              open(os.path.join(OUT, f"explain_{dataset}{'' if use_dcl else '_nodcl'}.json"), "w"))


if __name__ == "__main__":
    run(sys.argv[1], use_dcl=not (len(sys.argv) > 2 and sys.argv[2] == "nodcl"))
