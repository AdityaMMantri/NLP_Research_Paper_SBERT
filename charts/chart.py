"""
SBERT replication: Paper vs Replicated charts (Tables 1-5).
Run:  python make_charts.py   ->  PNG (300 dpi) + PDF per figure in ./charts
Requires: matplotlib, numpy

Conventions
- Paper values are exactly as printed in Reimers & Gurevych (2019), including the paper's own printed averages.
- Replicated values keep full precision internally and are labelled with 2 decimals (Table 4: 4 decimals, as in the paper).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT = "charts"
os.makedirs(OUT, exist_ok=True)

PAPER, REPL = "#2F5D8A", "#E8913A"
REPL_DARK = "#8C5A2B"
POS, NEG = "#3B8F5A", "#C0504D"
GRID, TXT, NOTE = "#D9D9D9", "#262626", "#595959"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.edgecolor": "#7F7F7F", "axes.labelcolor": TXT, "text.color": TXT,
    "xtick.color": TXT, "ytick.color": TXT, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})
LBL = 8.5   # data-label font size used on every chart


def style(ax, ylabel=None, xgrid=False):
    ax.grid(False)
    ax.grid(True, axis="x" if xgrid else "y", color=GRID, lw=0.8)
    if ylabel:
        ax.set_ylabel(ylabel)


def put_labels(ax, bars, texts, pad, size=LBL):
    for b, t in zip(bars, texts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad, t, ha="center", va="bottom", fontsize=size)


def fig_title(fig, title, top=0.90):
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.get_layout_engine().set(rect=(0, 0, 1, top))   # reserve room for title/legend above the axes


def footnote(fig, text):
    fig.text(0.5, -0.005, text, ha="center", va="top", fontsize=9, color=NOTE)


def pr_legend(fig, y=0.945):
    h = [Patch(color=PAPER, label="Paper (Reimers & Gurevych, 2019)"), Patch(color=REPL, label="Replicated")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y), ncol=2, fontsize=10.5)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("saved", name)


def f2(v):
    return f"{v:.2f}"


# ======================================================================
# DATA  (paper = printed values; replicated = full precision from notebooks)
# ======================================================================
t1_ds = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSb", "SICK-R"]
t1 = {  # name: (paper cols, paper printed avg, replicated cols)
    "SBERT-NLI-base": ([70.97, 76.53, 73.19, 79.09, 74.30, 77.03, 72.91], 74.89,
                       [70.664484, 72.778056, 70.662632, 78.869953, 74.058850, 76.779260, 73.363016]),
    "SRoBERTa-NLI-base": ([71.54, 72.49, 70.80, 78.74, 73.69, 77.77, 74.46], 74.21,
                          [72.614462, 73.975465, 71.350673, 79.862134, 76.457147, 77.776534, 73.395127]),
}

t2_paper = {"SBERT-STSb-base": (84.67, 0.19), "SRoBERTa-STSb-base": (84.92, 0.34),
            "SBERT-NLI-STSb-base": (85.35, 0.17), "SRoBERTa-NLI-STSb-base": (84.79, 0.38)}
t2_seeds = {
    "SBERT-STSb-base": [84.30618, 83.684361, 84.437164, 83.834628, 83.751215, 83.016717, 84.599774, 84.336687, 84.298234, 84.451104],
    "SRoBERTa-STSb-base": [84.844225, 85.357565, 85.015403, 85.089609, 85.212165, 85.382724, 85.054695, 85.359354, 85.215292, 84.890501],
    "SBERT-NLI-STSb-base": [84.471953, 84.323734, 84.778752, 84.267421, 84.54991, 84.21783, 84.284987, 84.552604, 84.845598, 84.803242],
    "SRoBERTa-NLI-STSb-base": [84.844225, 85.357565, 85.015403, 85.089609, 85.212165, 85.382724, 85.054695, 85.359354, 85.215292, 84.890501],
}

afs_paper = {"10-fold CV": (76.57, 74.13), "Cross-topic": (52.34, 50.65)}
folds_r = [75.764964, 77.309159, 76.064529, 77.456202, 78.751784, 72.867518, 77.438501, 74.057530, 75.674677, 73.328793]
folds_s = [73.902103, 74.557891, 72.612828, 74.415604, 75.791955, 70.729107, 76.221085, 71.631340, 72.339987, 69.857045]
top_r = [51.559500, 36.275879, 65.801347]
top_s = [50.834460, 32.615200, 64.158003]
afs_repl = {"10-fold CV": (float(np.mean(folds_r)), float(np.mean(folds_s))),
            "Cross-topic": (float(np.mean(top_r)), float(np.mean(top_s)))}

wiki_paper, wiki_repl = 0.8042, 0.7900088357844787

se_tasks = ["MR", "CR", "SUBJ", "MPQA", "SST", "TREC", "MRPC"]
se_paper_txt = ["83.64", "89.43", "94.39", "89.86", "88.96", "89.6", "76.00"]   # exactly as printed
se_paper = [float(v) for v in se_paper_txt]
se_paper_avg_txt, se_paper_avg = "87.41", 87.41                                   # printed average
se_repl = [82.14, 88.77, 93.09, 89.91, 88.41, 88.0, 76.23]

# ======================================================================
# FIG 1 - Table 1
# ======================================================================
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharey=True, constrained_layout=True)
x = np.arange(len(t1_ds) + 1)
w = 0.38
for ax, (name, (p, p_avg, r)) in zip(axes, t1.items()):
    p_all, r_all = p + [p_avg], r + [float(np.mean(r))]
    b1 = ax.bar(x - w / 2, p_all, w, color=PAPER)
    b2 = ax.bar(x + w / 2, r_all, w, color=REPL)
    put_labels(ax, b1, [f2(v) for v in p_all], 0.2)
    put_labels(ax, b2, [f2(v) for v in r_all], 0.2)
    ax.axvspan(len(t1_ds) - 0.5, len(t1_ds) + 0.5, color="#F2F2F2", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(t1_ds + ["Average"])
    ax.set_ylim(60, 84)
    ax.set_title(name, loc="left")
    style(ax, "Spearman correlation ×100")
fig_title(fig, "Table 1 — Unsupervised STS (cosine similarity, no STS training): Paper vs Replicated")
pr_legend(fig)
footnote(fig, "Y-axis truncated at 60. Paper averages are the values printed in the paper (74.89 / 74.21).")
save(fig, "fig1_table1_unsupervised_sts")

# ======================================================================
# FIG 2 - Table 2
# ======================================================================
names = list(t2_paper)
p_m = [t2_paper[n][0] for n in names]
p_s = [t2_paper[n][1] for n in names]
r_m = [float(np.mean(t2_seeds[n])) for n in names]
r_s = [float(np.std(t2_seeds[n], ddof=1)) for n in names]
fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
x = np.arange(len(names))
w = 0.36
b1 = ax.bar(x - w / 2, p_m, w, yerr=p_s, capsize=4, color=PAPER, ecolor="#1B1B1B")
b2 = ax.bar(x + w / 2, r_m, w, yerr=r_s, capsize=4, color=REPL, ecolor="#1B1B1B")
for b, m, s in list(zip(b1, p_m, p_s)) + list(zip(b2, r_m, r_s)):
    ax.text(b.get_x() + b.get_width() / 2, m + s + 0.05, f"{m:.2f}\n± {s:.2f}", ha="center", va="bottom", fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(82, 86.6)
style(ax, "STSb test Spearman ×100")
fig_title(fig, "Table 2 — STS Benchmark (supervised, 10 seeds): Paper vs Replicated")
pr_legend(fig)
footnote(fig, "Y-axis truncated at 82. Labels show mean ± std over 10 random seeds.")
save(fig, "fig2_table2_stsb_supervised")

# ======================================================================
# FIG 3 - Table 3
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), gridspec_kw={"width_ratios": [1.2, 1.35, 1.0]}, constrained_layout=True)
# (a)
ax = axes[0]
cats = ["10-fold CV\nPearson r", "10-fold CV\nSpearman ρ", "Cross-topic\nPearson r", "Cross-topic\nSpearman ρ"]
pv = [*afs_paper["10-fold CV"], *afs_paper["Cross-topic"]]
rv = [*afs_repl["10-fold CV"], *afs_repl["Cross-topic"]]
x = np.arange(4)
w = 0.38
b1 = ax.bar(x - w / 2, pv, w, color=PAPER)
b2 = ax.bar(x + w / 2, rv, w, color=REPL)
put_labels(ax, b1, [f2(v) for v in pv], 0.8)
put_labels(ax, b2, [f2(v) for v in rv], 0.8)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=10)
ax.set_ylim(0, 92)
ax.set_title("(a) Paper vs Replicated", loc="left")
style(ax, "Correlation ×100")
ax.legend(handles=[Patch(color=PAPER, label="Paper"), Patch(color=REPL, label="Replicated")],
          loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
# (b)
ax = axes[1]
f = np.arange(1, 11)
ax.plot(f, folds_r, "o-", color=REPL, lw=2, label="Replicated Pearson r")
ax.plot(f, folds_s, "s--", color=REPL_DARK, lw=2, label="Replicated Spearman ρ")
ax.axhline(afs_paper["10-fold CV"][0], color=PAPER, lw=1.8, label=f"Paper r = {afs_paper['10-fold CV'][0]:.2f}")
ax.axhline(afs_paper["10-fold CV"][1], color=PAPER, lw=1.8, ls=":", label=f"Paper ρ = {afs_paper['10-fold CV'][1]:.2f}")
ax.set_xticks(f)
ax.set_xlabel("Fold")
ax.set_ylim(66, 82)
ax.set_title("(b) 10-fold CV: per-fold replicated results", loc="left")
style(ax, "Correlation ×100")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
# (c)
ax = axes[2]
x = np.arange(3)
w = 0.38
b1 = ax.bar(x - w / 2, top_r, w, color=REPL)
b2 = ax.bar(x + w / 2, top_s, w, color=REPL_DARK)
put_labels(ax, b1, [f2(v) for v in top_r], 0.8)
put_labels(ax, b2, [f2(v) for v in top_s], 0.8)
ax.axhline(afs_paper["Cross-topic"][0], color=PAPER, lw=1.8)
ax.axhline(afs_paper["Cross-topic"][1], color=PAPER, lw=1.8, ls=":")
ax.set_xticks(x)
ax.set_xticklabels(["death\npenalty", "gay\nmarriage", "gun\ncontrol"])
ax.set_ylim(0, 92)
ax.set_title("(c) Cross-topic: per held-out topic", loc="left")
style(ax)
ax.legend(handles=[Patch(color=REPL, label="Replicated Pearson r"), Patch(color=REPL_DARK, label="Replicated Spearman ρ"),
                   Line2D([0], [0], color=PAPER, lw=1.8, label=f"Paper mean r = {afs_paper['Cross-topic'][0]:.2f}"),
                   Line2D([0], [0], color=PAPER, lw=1.8, ls=":", label=f"Paper mean ρ = {afs_paper['Cross-topic'][1]:.2f}")],
          loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
fig_title(fig, "Table 3 — Argument Facet Similarity (AFS), SBERT-AFS-base: Paper vs Replicated", top=0.94)
save(fig, "fig3_table3_afs")

# ======================================================================
# FIG 4 - Table 4
# ======================================================================
fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
bars = ax.bar(["Paper", "Replicated"], [wiki_paper, wiki_repl], width=0.42, color=[PAPER, REPL])
put_labels(ax, bars, [f"{wiki_paper:.4f}", f"{wiki_repl:.4f}"], 0.0012, size=12)
d = wiki_repl - wiki_paper
ax.text(0.5, 0.815, f"Replicated − Paper = {d:+.4f}", ha="center", fontsize=12, color=NEG if d < 0 else POS, fontweight="bold")
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0.76, 0.82)
style(ax, "Triplet accuracy")
ax.set_title("SBERT-WikiSec-base", loc="left")
fig_title(fig, "Table 4 — Wikipedia Section Triplets: Paper vs Replicated", top=0.94)
footnote(fig, f"Y-axis truncated at 0.76. Replicated exact value: {wiki_repl:.7f} (222,957 test triplets, 1 epoch, triplet loss).")
save(fig, "fig4_table4_wikisec")

# ======================================================================
# FIG 5 - Table 5
# ======================================================================
fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
tasks = se_tasks + ["Average"]
pp = se_paper + [se_paper_avg]
pp_txt = se_paper_txt + [se_paper_avg_txt]
rr = se_repl + [float(np.mean(se_repl))]
x = np.arange(len(tasks))
w = 0.38
b1 = ax.bar(x - w / 2, pp, w, color=PAPER)
b2 = ax.bar(x + w / 2, rr, w, color=REPL)
put_labels(ax, b1, pp_txt, 0.25, size=9)
put_labels(ax, b2, [f2(v) for v in rr], 0.25, size=9)
ax.axvspan(len(se_tasks) - 0.5, len(se_tasks) + 0.5, color="#F2F2F2", zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylim(65, 100)
style(ax, "Accuracy (%)")
ax.set_title("SBERT-NLI-base", loc="left")
fig_title(fig, "Table 5 — SentEval transfer tasks: Paper vs Replicated")
pr_legend(fig)
footnote(fig, "Y-axis truncated at 65. Paper labels are exactly as printed (TREC = 89.6).")
save(fig, "fig5_table5_senteval")

# ======================================================================
# FIG 6 - differences
# ======================================================================
rows = []
for n, (p, p_avg, r) in t1.items():
    rows.append((f"T1  {n} (avg)", float(np.mean(r)) - p_avg))
for n in names:
    rows.append((f"T2  {n}", float(np.mean(t2_seeds[n])) - t2_paper[n][0]))
rows += [("T3  AFS 10-fold (Pearson r)", afs_repl["10-fold CV"][0] - afs_paper["10-fold CV"][0]),
         ("T3  AFS cross-topic (Pearson r)", afs_repl["Cross-topic"][0] - afs_paper["Cross-topic"][0]),
         ("T4  WikiSec (accuracy ×100)", (wiki_repl - wiki_paper) * 100),
         ("T5  SentEval (avg)", float(np.mean(se_repl)) - se_paper_avg)]
labs, dv = [r[0] for r in rows], [r[1] for r in rows]
fig, ax = plt.subplots(figsize=(13, 6.6), constrained_layout=True)
y = np.arange(len(rows))[::-1]
ax.barh(y, dv, color=[POS if v >= 0 else NEG for v in dv], height=0.62)
ax.axvline(0, color="#404040", lw=1)
for yy, v in zip(y, dv):
    ax.text(v + (0.03 if v >= 0 else -0.03), yy, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=10)
ax.set_yticks(y)
ax.set_yticklabels(labs)
ax.set_xlim(min(dv) - 0.35, max(dv) + 0.35)
style(ax, xgrid=True)
ax.set_xlabel("Replicated − Paper (points on the ×100 scale)")
fig_title(fig, "Overall: how close is the replication to the paper?")
fig.legend(handles=[Patch(color=POS, label="Replicated ≥ paper"), Patch(color=NEG, label="Replicated < paper")],
           loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, fontsize=10.5)
save(fig, "fig6_summary_differences")
print("done ->", os.path.abspath(OUT))