"""Version-8 figures (matplotlib), rendered from results/*.json.
Usage: python figures_v8.py <results_dir> <out_dir>"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(os.path.join(R, n + ".json")))
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "lines.markersize": 3.6, "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
                     "pdf.fonttype": 42})
BLUE, VERM, GREEN, GREY, PURP, AMBER, SKY = "#0072B2", "#D55E00", "#009E73", "#595959", "#7B3294", "#E69F00", "#56B4E9"
STY = {"AX-NXGB": (BLUE, "o", "-"), "Sliding XGBoost": (VERM, "s", "--"), "ARF": (GREEN, "D", "-."), "Static XGBoost": (GREY, "^", ":")}
LAB = {"AX-NXGB": "AX-NXGB (proposed)", "Sliding XGBoost": "Sliding XGBoost", "ARF": "ARF", "Static XGBoost": "Static XGBoost"}


def block(v, b, k=3):
    v = np.asarray(v, float); w = k * b
    return np.array([v[i:i + w].mean() for i in range(0, len(v) - w + 1, b)])


def stream(ax, ds, b, title):
    r = J("main_" + ds)["methods"]
    for m in STY:
        a = r[m]["runs"][0]["acc_chunks"]; c, mk, ls = STY[m]
        y = block(a, b); x = (block(np.arange(len(a)), b) + 1.5) * 0.25
        ax.plot(x, 100 * y, ls, color=c, marker=mk, label=LAB[m], mfc=c if m == "AX-NXGB" else "white", mew=0.8)
    al = np.where(J("explain_" + ds)["log"]["alarm"])[0]
    ax.plot((al + 0.5) * 0.25, np.full(len(al), 31.5), "|", color=PURP, ms=6, mew=0.9, label="ADWIN alarm")
    ax.set_ylim(30, 100); ax.set_xlabel(r"Processed samples ($\times 10^3$)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=8, pad=3); ax.grid(alpha=0.25, lw=0.4)


fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.35), gridspec_kw=dict(wspace=0.38))
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.17, top=0.78)
stream(axs[0], "electricity", 8, "(a) Electricity")
stream(axs[1], "covertype", 16, "(b) Covertype")
axs[0].set_xlim(0, 46); axs[1].set_xlim(0, 100)
h, l = axs[0].get_legend_handles_labels()
x0 = axs[0].get_position().x0; x1 = axs[1].get_position().x1
LEG = fig.legend(h, l, loc="lower left", ncol=3, mode="expand", bbox_to_anchor=(x0, 0.86, x1 - x0, 0.12), fontsize=6.4, handlelength=1.7, columnspacing=0.5, handletextpad=0.4, borderpad=0.3, labelspacing=0.25, borderaxespad=0)
t = J("tradeoff_covertype")["methods"]; Rs = [1, 2, 4, 8, 16, 32, 64]; ax = axs[2]
for kind, lab, (c, mk, ls) in [("hybrid", "AX-NXGB (hybrid)", STY["AX-NXGB"]), ("tree", "Tree branch only", STY["Sliding XGBoost"])]:
    x = [t[f"{kind}|R={k}"]["mean"]["n_refit"] for k in Rs]; y = [100 * t[f"{kind}|R={k}"]["mean"]["acc"] for k in Rs]
    ax.plot(x, y, ls, color=c, marker=mk, label=lab, mfc=c if kind == "hybrid" else "white", mew=0.8)
a = t["hybrid|adaptive"]["mean"]
ax.plot([a["n_refit"]], [100 * a["acc"]], "p", color=BLUE, mfc="white", ms=6, mew=1.0, label="AX-NXGB-Lite")
ax.set_ylim(58, 97); ax.set_xscale("log"); ax.set_xlabel("Tree refits"); ax.set_ylabel("Accuracy (%)")
ax.set_title("(c) Covertype: accuracy vs. refits", fontsize=8, pad=3); ax.grid(alpha=0.25, lw=0.4, which="both")
ax.legend(loc="lower right", fontsize=6.2, handlelength=1.8, borderpad=0.3, labelspacing=0.2)
# importance shares, Electricity
e = J("explain_electricity"); L = e["log"]; names = e["features"]
I = np.array(L["I_dyn"]); S = I / I.sum(1, keepdims=True); b = 8
x = block(L["chunk"], b) * 0.25
feats = [("nswprice", "NSW price", BLUE, "o", "-"), ("nswdemand", "NSW demand", VERM, "s", "--"), ("period", "Time of day", GREEN, "D", "-."),
         ("vicprice", "VIC price", PURP, "^", ":"), ("vicdemand", "VIC demand", SKY, "v", "-"), ("transfer", "Transfer", AMBER, "x", "--")]
ax = axs[3]
for f, lab, c, mk, ls in feats:
    y = block(S[:, names.index(f)], b); k = y > 0
    ax.plot(x[k], 100 * y[k], ls, color=c, marker=mk, label=lab, mfc="white" if mk not in "x" else c, mew=0.8)
ax.axvline(17.424, color="0.4", lw=0.6, ls=(0, (2, 2)))

ax.set_yscale("log"); ax.set_ylim(0.1, 30000); ax.set_xlim(0, 46)
ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1", "1", "10", "100"]); ax.minorticks_off()
ax.set_xlabel(r"Processed samples ($\times 10^3$)"); ax.set_ylabel(r"Share of mean $|\Phi_j|$ (%)", labelpad=1); ax.set_title("(d) Electricity: importance", fontsize=8, pad=3)
ax.grid(alpha=0.25, lw=0.4); ax.legend(loc="upper center", ncol=2, fontsize=5.6, handlelength=1.6, columnspacing=0.5, borderpad=0.2, labelspacing=0.1)
fig.savefig(os.path.join(OUT, "fig_results_v5.pdf"), bbox_inches="tight", pad_inches=0.02)
