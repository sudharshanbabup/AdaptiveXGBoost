"""Chunk-wise prequential (test-then-train) evaluation shared by all experiments."""
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score


def prequential(model, X, y, chunk=250, warm=1, hook=None):
    """warm = number of initial chunks used only for training (not scored)."""
    n = len(y)
    P = np.full(n, np.nan)
    t_pred = t_learn = 0.0
    for c, s in enumerate(range(0, n, chunk)):
        e = min(s + chunk, n)
        Xc, yc = X[s:e], y[s:e]
        t0 = time.perf_counter(); p = model.predict_proba(Xc); t_pred += time.perf_counter() - t0
        P[s:e] = p
        err = ((p >= 0.5).astype(int) != yc).astype(int)
        if hook is not None:
            hook(c, s, e, model, Xc, yc, p, phase="pre")
        t0 = time.perf_counter(); model.learn(Xc, yc, err); t_learn += time.perf_counter() - t0
        if hook is not None:
            hook(c, s, e, model, Xc, yc, p, phase="post")
    s0 = warm * chunk
    return dict(P=P, start=s0, t_pred=t_pred, t_learn=t_learn)


def summarize(P, y, start, chunk=250):
    yt, p = y[start:], P[start:]
    yh = (p >= 0.5).astype(int)
    acc_chunks = np.array([accuracy_score(yt[i:i + chunk], yh[i:i + chunk]) for i in range(0, len(yt), chunk)])
    return dict(acc=accuracy_score(yt, yh), f1=f1_score(yt, yh, average="macro"),
                kappa=cohen_kappa_score(yt, yh), auc=roc_auc_score(yt, p),
                acc_chunks=acc_chunks, worst10=float(np.sort(acc_chunks)[:max(1, len(acc_chunks) // 10)].mean()))
