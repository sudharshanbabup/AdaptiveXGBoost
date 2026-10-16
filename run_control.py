"""
Controlled Electricity experiments (version 5) for the label-construction / leakage question.
  orig     : benchmark label (UP/DOWN vs. 24-h moving average), all 7 features      (reproduction check)
  noprice  : benchmark label, NSW price REMOVED from the inputs (6 features)
  next     : alternative target y_i = 1[nswprice_{i+1} > nswprice_i] (next half-hour price rise),
             inputs x_i at time i only (last record dropped). Not defined via a moving average.
Usage: python run_control.py <setting>   -> results/control_<setting>.json
"""
import sys, json, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from data import DATA_DIR
from models import AdaptiveNeuralXGB, WindowXGB, RiverWrap, DCLWrap
from stream_eval import prequential, summarize
CHUNK, H, SEEDS = 250, 6000, 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def load(setting):
    df = pd.read_csv(os.path.join(DATA_DIR, "electricity.csv"))
    X = df.drop(columns=["class", "date"])           # day, period, nswprice, nswdemand, vicprice, vicdemand, transfer
    if setting == "next":
        pr = df["nswprice"].values
        y = (pr[1:] > pr[:-1]).astype(int); X = X.iloc[:-1]
    else:
        y = (df["class"].astype(str).str.upper() == "UP").astype(int).values
    if setting == "noprice":
        X = X.drop(columns=["nswprice"])
    cont = [i for i, c in enumerate(X.columns) if c != "day"]
    return X.values.astype(np.float32), y, list(X.columns), cont


def run(setting):
    X, y, names, cont = load(setting); p = X.shape[1]; nd = p + 3 * len(cont)
    M = {"AX-NXGB": lambda s: AdaptiveNeuralXGB(p, seed=s, horizon=H, dcl_cols=cont, refit_every=1),
         "AX-NXGB w/o DCL": lambda s: AdaptiveNeuralXGB(p, seed=s, horizon=H, dcl_cols=cont, refit_every=1, use_dcl=False),
         "Sliding XGBoost": lambda s: WindowXGB(p, s, window=H),
         "Sliding XGBoost + DCL": lambda s: DCLWrap(WindowXGB(nd, s, window=H), p, cols=cont),
         "ARF": lambda s: RiverWrap(p, s, "arf")}
    res = dict(setting=setting, n=int(len(y)), p=p, features=names, pos_rate=float(y.mean()),
               nochange_sample=float(np.mean(y[CHUNK:] == y[CHUNK - 1:-1])), methods={})
    for name, make in M.items():
        rows = []
        for s in range(SEEDS):
            t0 = time.time(); m = make(s); r = prequential(m, X, y, CHUNK); sm = summarize(r["P"], y, r["start"], CHUNK)
            rows.append(dict(seed=s, acc=sm["acc"], f1=sm["f1"], kappa=sm["kappa"], wall=time.time() - t0))
            print(f"[{setting}] {name:24s} seed={s} acc={sm['acc']:.4f} f1={sm['f1']:.4f} t={time.time()-t0:.0f}s", flush=True)
        res["methods"][name] = dict(runs=rows, mean={k: float(np.mean([r[k] for r in rows])) for k in ("acc", "f1", "kappa")},
                                    std={k: float(np.std([r[k] for r in rows], ddof=1)) for k in ("acc", "f1", "kappa")})
        json.dump(res, open(os.path.join(OUT, f"control_{setting}.json"), "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1])
