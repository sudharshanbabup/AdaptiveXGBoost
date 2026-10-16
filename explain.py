"""
Dynamic SHAP for the hybrid model  F_t(a) = alpha_t T_t(a) + gamma_t s_theta(a),  a = [x, c],
reported in the RAW input space, plus the Explanation Drift Index (EDI).

  tree branch   : interventional TreeSHAP w.r.t. the reference set B_t      (exact for T_t)
  neural branch : Aumann-Shapley (path-integrated) values averaged over B_t (complete for s)
  linearity     : phi^F = alpha * phi^T + gamma * phi^s
  grouping      : feature j owns x_j and its DCL innovations c_j^(1..S)  ->  Phi_j
  => efficiency : sum_j Phi_j = F_t(a) - E_{b~B_t}[F_t(b)]
"""
import itertools
from math import factorial
import numpy as np
import torch
import shap


def feature_groups(p, dcl):
    """index sets G_j (columns of a owned by raw feature j)."""
    G = [[j] for j in range(p)]
    if dcl is not None:
        for s in range(dcl.S):
            for k, j in enumerate(dcl.cols):
                G[int(j)].append(p + s * dcl.p + k)
    return G


def tree_shap(model, A, B):
    if model.clf is None:
        return np.zeros_like(A, dtype=float)
    ex = shap.TreeExplainer(model.clf, data=B, feature_perturbation="interventional", model_output="raw")
    return np.asarray(ex.shap_values(A, check_additivity=False), dtype=float)


def neural_shap(model, A, B, steps=16):
    """Expected path-integrated gradients of s_theta w.r.t. a, averaged over the reference set."""
    if not model.use_neural or model.enc.mu is None:
        return np.zeros_like(A, dtype=float)
    enc = model.enc
    At, Bt = enc.scale(A), enc.scale(B)                         # standardised coordinates
    taus = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    out = torch.zeros_like(At)
    for b in Bt:                                               # loop over reference points
        diff = At - b                                          # n x pa
        pts = (b + taus[:, None, None] * diff[None]).reshape(-1, At.shape[1]).requires_grad_(True)
        s = enc.net(pts)[2].sum()
        g = torch.autograd.grad(s, pts)[0].reshape(steps, *At.shape).mean(0)
        out += diff * g
    return (out / len(Bt)).numpy().astype(float)


def hybrid_shap(model, A, B, groups):
    """Input-space attributions Phi (n x p) and the reference value E_B[F]."""
    phi = model.alpha * tree_shap(model, A, B) + model.gamma * neural_shap(model, A, B)
    Phi = np.stack([phi[:, g].sum(1) for g in groups], 1)
    return Phi, float(model.margin(B).mean())


def exact_group_shapley(model, A, B, groups):
    """Brute-force interventional Shapley values of the END-TO-END model with the p raw
    features as players (feature j is swapped together with its innovations)."""
    p, n, nb = len(groups), len(A), len(B)
    val = {}
    for r in range(p + 1):
        for S in itertools.combinations(range(p), r):
            cols = [c for j in S for c in groups[j]]
            M = np.repeat(B[None], n, 0).copy()                 # n x nb x pa
            if cols:
                M[:, :, cols] = A[:, None, cols]
            val[S] = model.margin(M.reshape(n * nb, -1)).reshape(n, nb).mean(1)
    Phi = np.zeros((n, p))
    for j in range(p):
        others = [k for k in range(p) if k != j]
        for r in range(p):
            w = factorial(r) * factorial(p - r - 1) / factorial(p)
            for S in itertools.combinations(others, r):
                Phi[:, j] += w * (val[tuple(sorted(S + (j,)))] - val[S])
    return Phi


# ------------------------------------------------------------------ explanation-drift metrics
def importance_profile(Phi):
    """global importance vector  I = mean_i |Phi_ij|."""
    return np.abs(Phi).mean(0)


def edi(I0, I1, eps=1e-12):
    """EDI = 1 - cos(I0, I1) = 0.5 * ||u0 - u1||^2 for unit-normalised profiles; also returns
    the exact per-feature decomposition delta_j = 0.5 (u0_j - u1_j)^2  (sum_j delta_j = EDI).
    v4: if either profile is (numerically) zero, u is undefined and EDI is returned as NaN
    (paper, Sec. IV-C). This case did not occur in any explained chunk of the reported runs."""
    n0, n1 = np.linalg.norm(I0), np.linalg.norm(I1)
    if n0 <= eps or n1 <= eps:
        return float("nan"), np.full(len(I0), np.nan)
    u0, u1 = I0 / n0, I1 / n1
    d = 0.5 * (u0 - u1) ** 2
    return float(d.sum()), d


def cosine_rows(P, Q, eps=1e-12):
    return (P * Q).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(Q, axis=1) + eps)
