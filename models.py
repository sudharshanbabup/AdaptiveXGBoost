"""
Adaptive Explainable Neural-XGBoost (AX-NXGB) and the streaming baselines.

Every learner exposes
    predict_proba(X) -> P(y=1)            (test step of the prequential protocol)
    learn(X, y, err)                      (train step; err = 0/1 errors made on this chunk)
so that all methods are evaluated with exactly the same test-then-train loop.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from river.drift import ADWIN

torch.set_num_threads(1)
XGB_PARAMS = dict(n_estimators=100, max_depth=5, learning_rate=0.15, subsample=0.9,
                  colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist",
                  n_jobs=1, verbosity=0)


def make_xgb(seed=0, **kw):
    p = dict(XGB_PARAMS); p.update(kw)
    return xgb.XGBClassifier(random_state=seed, **p)


def _safe_fit(clf, X, y, w=None):
    """XGBoost needs both classes; return None if the window is single-class."""
    if len(np.unique(y)) < 2:
        return None
    clf.fit(X, y, sample_weight=w)
    return clf


# --------------------------------------------------------------------------- neural part
class SupAE(nn.Module):
    """Supervised auto-encoder: encoder f_theta, decoder g_psi and a linear class head."""

    def __init__(self, p, d=8, h=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(p, h), nn.ReLU(), nn.Linear(h, d), nn.Tanh())
        self.dec = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, p))
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z), self.head(z).squeeze(-1)


class OnlineEncoder:
    """Encoder trained online with L = BCE + lam * MSE; inputs are standardised with
    exponentially-forgotten running statistics so that scaling also follows the stream."""

    def __init__(self, p, d=8, h=32, lr=2e-3, lam=0.5, epochs=3, bs=64, seed=0, momentum=0.05):
        torch.manual_seed(seed)
        self.p, self.d = p, d
        self.net = SupAE(p, d, h)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        self.lr0 = lr
        self.lam, self.epochs, self.bs = lam, epochs, bs
        self.mu, self.sd, self.momentum = None, None, momentum
        self.rng = np.random.default_rng(seed)

    def _update_scaler(self, X):
        m, s = X.mean(0), X.std(0) + 1e-6
        if self.mu is None:
            self.mu, self.sd = m, s
        else:
            a = self.momentum
            self.mu = (1 - a) * self.mu + a * m
            self.sd = (1 - a) * self.sd + a * s

    def scale(self, X):
        return torch.as_tensor(np.array((X - self.mu) / self.sd), dtype=torch.float32)

    def fit(self, X, y, epochs=None, lr_scale=1.0, base=None):
        self._update_scaler(X)
        for g in self.opt.param_groups:
            g["lr"] = self.lr0 * lr_scale
        Xt, yt = self.scale(X), torch.as_tensor(np.array(y), dtype=torch.float32)
        n = len(X)
        bt = torch.zeros(n) if base is None else torch.as_tensor(np.array(base), dtype=torch.float32)
        self.net.train()
        for _ in range(epochs or self.epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, self.bs):
                idx = perm[i:i + self.bs]
                z, xr, lg = self.net(Xt[idx])
                loss = nn.functional.binary_cross_entropy_with_logits(lg + bt[idx], yt[idx]) \
                    + self.lam * nn.functional.mse_loss(xr, Xt[idx])
                self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.net.eval()

    @torch.no_grad()
    def transform(self, X):
        return self.net.enc(self.scale(X)).numpy()

    @torch.no_grad()
    def proba(self, X):
        return torch.sigmoid(self.net(self.scale(X))[2]).numpy()


# --------------------------------------------------------------------------- proposed
class DriftCompensation:
    """
    Label-free drift-compensation layer (DCL).
    For every scale beta_s it tracks an exponentially-forgotten mean/variance of each raw
    feature and emits the *innovation* c = (x - m) / sqrt(v) computed with the statistics
    available BEFORE x arrives (strictly causal, no labels needed).
    Output ordering: [scale 1: p features | scale 2: p features | ...].
    """

    def __init__(self, p, betas=(0.005, 0.02, 0.1), clip=8.0, cols=None):
        self.cols = np.arange(p) if cols is None else np.asarray(cols)      # continuous features only
        p = len(self.cols)
        self.p, self.betas, self.clip = p, np.asarray(betas, float)[:, None], clip
        self.m = None
        self.v = None

    @property
    def S(self):
        return len(self.betas)

    def transform_update(self, X):
        X = np.asarray(X)[:, self.cols]
        n = len(X)
        out = np.empty((n, self.S * self.p), np.float32)
        if self.m is None:
            self.m = np.repeat(X[:1].astype(float), self.S, 0)
            self.v = np.tile(np.maximum(X[:max(2, min(50, n))].var(0), 1e-6), (self.S, 1))
        b = self.betas
        for t in range(n):
            d = X[t][None, :] - self.m
            out[t] = np.clip(d / np.sqrt(self.v + 1e-10), -self.clip, self.clip).ravel()
            self.m += b * d
            self.v = (1 - b) * (self.v + b * d * d)
        return out


class AdaptiveNeuralXGB:
    """
    Proposed AX-NXGB (Adaptive eXplainable Neural-XGBoost).

        F_t(x) = alpha_t * T_t(a) + gamma_t * s_theta(a),      a = [x, c]

        c        drift-compensated innovations of x (DCL, label-free, causal)
        T_t      XGBoost margin, refit on the last `horizon` samples every `refit_every`
                 chunks AND whenever ADWIN raises an alarm on the 0/1 error stream
        s_theta  margin of an online supervised auto-encoder (updated every chunk):
                 the fast path that keeps tracking the concept while the trees are stale
        (alpha_t, gamma_t)  drift gate = held-out (prequential) log-loss minimiser over the
                 last `gate_mem` chunks; it shifts trust between the two branches
    Ablation switches: use_dcl, use_neural, gate, refit_on_drift, tree_latent, latent_only.
    """

    def __init__(self, p, d=8, seed=0, horizon=6000, delta=0.002, betas=(0.005, 0.02, 0.1),
                 use_dcl=True, dcl_cols=None, use_neural=True, gate=True, gate_mem=4, drift_boost=1.0,
                 refit_every=8, refit_on_drift=True, tree_latent=False, latent_only=False, r_max=8):
        self.p, self.d, self.seed, self.r_max = p, d, seed, r_max
        self.use_dcl, self.use_neural = use_dcl, use_neural
        self.gate = gate and use_neural
        self.horizon, self.gate_mem, self.drift_boost = horizon, gate_mem, drift_boost
        self.refit_every, self.refit_on_drift = refit_every, refit_on_drift
        self.latent_only = latent_only and use_neural            # v2 design: trees see z only
        self.tree_latent = (tree_latent or latent_only) and use_neural
        self.dcl = DriftCompensation(p, betas, cols=dcl_cols) if use_dcl else None
        self.pa = p + (self.dcl.S * self.dcl.p if use_dcl else 0)  # dim of a
        self.enc = OnlineEncoder(self.pa, d, seed=seed) if use_neural else None
        self.snap = None                                         # encoder snapshot used by the trees
        self.adwin = ADWIN(delta=delta)
        self.Ab = np.empty((0, self.pa), np.float32); self.yb = np.empty(0, int)
        self.clf, self.hist = None, []
        self.alpha, self.gamma = 1.0, (1.0 if use_neural else 0.0)
        if self.latent_only:
            self.gamma = 0.0; self.gate = False
        self.t = self.chunk_idx = self.n_refit = 0
        self.drift_log, self.alpha_log, self.gamma_log, self.refit_log = [], [], [], []
        self.last, self.drift_flag, self.refit_flag = None, False, False

    # ---- feature pipeline -------------------------------------------------------------
    def hybrid_input(self, X):
        """a = [x, c]; advances the DCL state (call once per sample, in stream order)."""
        return np.hstack([X, self.dcl.transform_update(X)]) if self.use_dcl else np.asarray(X, np.float32)

    def _latent(self, A):
        net, mu, sd = self.snap
        with torch.no_grad():
            return net.enc(torch.as_tensor(np.array((A - mu) / sd), dtype=torch.float32)).numpy()

    def tree_input(self, A):
        if self.latent_only:
            return self._latent(A)
        return np.hstack([A, self._latent(A)]) if self.tree_latent else A

    def neural_margin(self, A):
        if not self.use_neural or self.enc.mu is None:
            return np.zeros(len(A), np.float32)
        with torch.no_grad():
            return self.enc.net(self.enc.scale(A))[2].numpy()

    def tree_margin(self, A):
        if self.clf is None:
            return np.zeros(len(A), np.float32)
        return self.clf.predict(self.tree_input(A), output_margin=True)

    def margin(self, A):
        """F_t(a) for already-compensated inputs (stateless; used by the explainer)."""
        return self.alpha * self.tree_margin(A) + self.gamma * self.neural_margin(A)

    def predict_proba(self, X):
        A = self.hybrid_input(X)
        T, s = self.tree_margin(A), self.neural_margin(A)
        self.last = dict(A=A, T=T, s=s)
        if self.clf is None:
            return 1.0 / (1.0 + np.exp(-s))
        return 1.0 / (1.0 + np.exp(-(self.alpha * T + self.gamma * s)))

    # ---- learning ---------------------------------------------------------------------
    def _update_gate(self, y):
        if not self.gate or self.clf is None:
            return
        self.hist.append((self.last["T"], self.last["s"], y)); self.hist = self.hist[-self.gate_mem:]
        T = np.concatenate([h[0] for h in self.hist]); s = np.concatenate([h[1] for h in self.hist])
        yy = 2 * np.concatenate([h[2] for h in self.hist]) - 1
        g = np.linspace(0, 1.2, 13)
        L = np.array([[np.mean(np.logaddexp(0, -yy * (a * T + c * s))) for c in g] for a in g])
        i, j = np.unravel_index(np.argmin(L), L.shape)
        self.alpha, self.gamma = float(g[i]), float(g[j])

    def _schedule(self, drift):
        """Refit period R_t: reset to 1 on an ADWIN alarm, doubled (up to r_max) after every
        quiet refit -> frequent refits only while the concept is moving."""
        if not hasattr(self, "period"):
            self.period, self.wait = 1, 0
        if drift:
            self.period, self.wait = 1, 0
            return True
        self.wait += 1
        if self.clf is None or self.wait >= self.period:
            self.period, self.wait = min(2 * self.period, self.r_max), 0
            return True
        return False

    def learn(self, X, y, err):
        A = self.last["A"]
        # (i) drift gate from held-out margins of this chunk
        self._update_gate(y)
        self.alpha_log.append(self.alpha); self.gamma_log.append(self.gamma)
        # (ii) ADWIN on the error stream
        drift = False
        for e in err:
            self.adwin.update(float(e)); drift |= self.adwin.drift_detected
        self.drift_flag = drift
        if drift:
            self.drift_log.append(self.t)
        self.t += len(X)
        # (iii) fast path: encoder update every chunk (larger step after an alarm)
        if self.use_neural:
            self.enc.fit(A, y, lr_scale=self.drift_boost if drift else 1.0,
                         epochs=self.enc.epochs * (2 if drift else 1))
        # (iv) slow path: periodic + drift-triggered tree refit on the horizon
        self.Ab = np.vstack([self.Ab, A])[-self.horizon:]
        self.yb = np.concatenate([self.yb, y])[-self.horizon:]
        if self.refit_every == "adaptive":             # drift-adaptive schedule (see _schedule)
            due = self._schedule(drift)
        else:
            due = (self.clf is None) or (self.chunk_idx % self.refit_every == 0) or (drift and self.refit_on_drift)
        self.chunk_idx += 1
        self.refit_flag = False
        if due and len(np.unique(self.yb)) > 1:
            if self.tree_latent:
                self.snap = (copy.deepcopy(self.enc.net), self.enc.mu.copy(), self.enc.sd.copy())
            self.clf = make_xgb(self.seed).fit(self.tree_input(self.Ab), self.yb)
            self.n_refit += 1; self.refit_flag = True
        self.refit_log.append(self.refit_flag)


class DCLWrap:
    """Any baseline fed with [x, c] (used to isolate the contribution of the DCL)."""

    def __init__(self, base, p, betas=(0.005, 0.02, 0.1), cols=None):
        self.base, self.dcl, self._A = base, DriftCompensation(p, betas, cols=cols), None

    def predict_proba(self, X):
        self._A = np.hstack([X, self.dcl.transform_update(X)])
        return self.base.predict_proba(self._A)

    def learn(self, X, y, err):
        self.base.learn(self._A, y, err)


# --------------------------------------------------------------------------- baselines
class StaticXGB:
    """XGBoost trained once on the first n0 samples (the usual offline deployment)."""

    def __init__(self, p, seed=0, n0=2000):
        self.seed, self.n0 = seed, n0
        self.X, self.y, self.clf, self.done = [], [], None, False

    def predict_proba(self, X):
        return np.full(len(X), 0.5) if self.clf is None else self.clf.predict_proba(X)[:, 1]

    def learn(self, X, y, err):
        if self.done:
            return
        self.X.append(X); self.y.append(y)
        Xa, ya = np.vstack(self.X), np.concatenate(self.y)
        self.clf = _safe_fit(make_xgb(self.seed), Xa, ya) or self.clf
        if len(ya) >= self.n0:
            self.done = True; self.X = self.y = None


class WindowXGB:
    """XGBoost re-trained every chunk on a fixed sliding window."""

    def __init__(self, p, seed=0, window=2000, booster="xgb"):
        self.seed, self.window, self.booster = seed, window, booster
        self.Xb = np.empty((0, p), np.float32); self.yb = np.empty(0, int); self.clf = None

    def _make(self):
        if self.booster == "lgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.15, subsample=0.9,
                                      subsample_freq=1, colsample_bytree=0.9, num_leaves=31,
                                      random_state=self.seed, n_jobs=2, verbose=-1)
        return make_xgb(self.seed)

    def predict_proba(self, X):
        return np.full(len(X), 0.5) if self.clf is None else self.clf.predict_proba(X)[:, 1]

    def learn(self, X, y, err):
        self.Xb = np.vstack([self.Xb, X])[-self.window:]
        self.yb = np.concatenate([self.yb, y])[-self.window:]
        if len(np.unique(self.yb)) > 1:
            self.clf = self._make().fit(self.Xb, self.yb)


class AdwinResetXGB:
    """Detect-and-retrain: XGBoost on a growing buffer that is flushed on an ADWIN alarm."""

    def __init__(self, p, seed=0, delta=0.002, w_max=6000):
        self.seed, self.w_max = seed, w_max
        self.adwin = ADWIN(delta=delta)
        self.Xb = np.empty((0, p), np.float32); self.yb = np.empty(0, int); self.clf = None

    def predict_proba(self, X):
        return np.full(len(X), 0.5) if self.clf is None else self.clf.predict_proba(X)[:, 1]

    def learn(self, X, y, err):
        drift = False
        for e in err:
            self.adwin.update(float(e)); drift |= self.adwin.drift_detected
        if drift:
            self.Xb = np.empty((0, X.shape[1]), np.float32); self.yb = np.empty(0, int)
        self.Xb = np.vstack([self.Xb, X])[-self.w_max:]
        self.yb = np.concatenate([self.yb, y])[-self.w_max:]
        c = _safe_fit(make_xgb(self.seed), self.Xb, self.yb)
        if c is not None:
            self.clf = c


class IncrementalXGB:
    """Continued boosting: a few new trees per chunk, oldest trees dropped (AXGB-style push)."""

    def __init__(self, p, seed=0, trees_per_chunk=10, max_trees=100):
        self.seed, self.k, self.max_trees = seed, trees_per_chunk, max_trees
        self.bst = None
        self.params = dict(objective="binary:logistic", max_depth=5, eta=0.15, subsample=0.9,
                           colsample_bytree=0.9, tree_method="hist", nthread=2, verbosity=0, seed=seed)

    def predict_proba(self, X):
        if self.bst is None:
            return np.full(len(X), 0.5)
        return self.bst.predict(xgb.DMatrix(X))

    def learn(self, X, y, err):
        if len(np.unique(y)) < 2 and self.bst is None:
            return
        self.bst = xgb.train(self.params, xgb.DMatrix(X, label=y), num_boost_round=self.k, xgb_model=self.bst)
        n = self.bst.num_boosted_rounds()
        if n > self.max_trees:
            self.bst = self.bst[n - self.max_trees:n]


class OnlineMLP:
    """The neural branch alone (online supervised auto-encoder head)."""

    def __init__(self, p, seed=0):
        self.enc = OnlineEncoder(p, seed=seed); self.ready = False

    def predict_proba(self, X):
        return self.enc.proba(X) if self.ready else np.full(len(X), 0.5)

    def learn(self, X, y, err):
        self.enc.fit(X, y); self.ready = True


class RiverWrap:
    """Instance-incremental river learners (ARF, Hoeffding Adaptive Tree, ...)."""

    def __init__(self, p, seed=0, kind="arf"):
        from river import forest, tree, preprocessing, linear_model, ensemble
        if kind == "arf":
            self.m = forest.ARFClassifier(n_models=10, seed=seed)
        elif kind == "hat":
            self.m = tree.HoeffdingAdaptiveTreeClassifier(seed=seed)
        elif kind == "srp":
            self.m = ensemble.SRPClassifier(n_models=10, seed=seed)
        self.p = p

    def predict_proba(self, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            out[i] = self.m.predict_proba_one(dict(enumerate(x))).get(1, 0.5)
        return out

    def learn(self, X, y, err):
        for x, yy in zip(X, y):
            self.m.learn_one(dict(enumerate(x)), int(yy))
