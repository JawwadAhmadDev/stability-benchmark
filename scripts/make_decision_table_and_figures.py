"""Rebuild the decision table and all eight figures of the paper from the released results.

Input : results/nb5_master_long_form_30x4.csv   (written by NB4; one row per metric x method x dataset)
Output: results/decision_table_30methods.csv
        figures/Fig1_methodology.{png,pdf}
        figures/Fig2_CD_diagrams.{png,pdf}
        figures/Fig4_family_ranks.{png,pdf}
        figures/Fig3_stability_by_method.{png,pdf}
        figures/Fig5_performance_vs_stability.{png,pdf}
        figures/Fig6_outcome_correlations.{png,pdf}
        figures/Fig7_stability_rank_per_dataset.{png,pdf}
        figures/Fig8_nogueira_vs_jaccard.{png,pdf}

Run from the repository root:  python scripts/make_decision_table_and_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.font_manager import FontProperties
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG = os.path.join(ROOT, 'results', 'nb5_master_long_form_30x4.csv')
TABLE_OUT = os.path.join(ROOT, 'results', 'decision_table_30methods.csv')
FIG_DIR = os.path.join(ROOT, 'figures')
DPI = 600
ALPHA = 0.05

COL = {'Filter': '#2f5d9e', 'Embedded': '#0e7c7b', 'Hybrid': '#c08a2e', 'Wrapper': '#c1466b'}
SHORT = {'Hybrid/Advanced': 'Hybrid', 'Wrapper/Metaheuristic': 'Wrapper'}


# ----------------------------------------------------------------------------- data and ranks
def load():
    L = pd.read_csv(LONG)
    mats = {m: L[L.metric == m].pivot(index='method', columns='dataset', values='value') for m in L.metric.unique()}
    family = L.drop_duplicates('method').set_index('method')['family']
    return mats, family


def average_ranks(mat, higher_is_better=True):
    """Rank the methods on each dataset (1 = best, ties averaged) and average across datasets."""
    return mat.rank(axis=0, ascending=not higher_is_better, method='average').mean(axis=1)


def nemenyi_cd(k, n, alpha=ALPHA):
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


# ----------------------------------------------------------------------------- decision table
def decision_table(mats, family):
    stab = average_ranks(mats['Nogueira'], True)
    comp = average_ranks(mats['Compactness'], False)
    auc = average_ranks(mats['AUC'], True)
    t = pd.DataFrame({
        'method': stab.index,
        'stab_rank': stab.values,
        'comp_rank': comp.reindex(stab.index).values,
        'auc_rank': auc.reindex(stab.index).values,
        'mean_Nog': mats['Nogueira'].mean(axis=1).reindex(stab.index).values,
        'mean_AUC': mats['AUC'].mean(axis=1).reindex(stab.index).values,
        'mean_feat': mats['Compactness'].mean(axis=1).reindex(stab.index).values,
    })
    t['score_w0.34'] = t['stab_rank'] + 0.34 * t['comp_rank']
    t['score_w0.20'] = t['stab_rank'] + 0.20 * t['comp_rank']
    t['score_stab_only'] = t['stab_rank']          # stability first ...
    t['family'] = t['method'].map(family)
    t = t.sort_values(['score_stab_only', 'comp_rank'], kind='mergesort').reset_index(drop=True)  # ... compactness breaks ties
    t.to_csv(TABLE_OUT, index=False)
    # weight at which compactness would change the selected method
    s, c = t.set_index('method')['stab_rank'], t.set_index('method')['comp_rank']
    top = t.loc[0, 'method']
    cross = {m: (s[m] - s[top]) / (c[top] - c[m]) for m in s.index if m != top and c[m] < c[top]}
    m_first = min(cross, key=cross.get)
    print(f'Decision table -> {TABLE_OUT}')
    print(f'  selected method: {top}; it stays selected for compactness weights w < {cross[m_first]:.3f} ({m_first} above)')
    return t


# ----------------------------------------------------------------------------- Figure 1 (methodology)
def figure1():
    FONT = 'DejaVu Sans'; W = 7.16
    TITLE, HEAD, BODY, BAR = 8.5, 7.5, 7.0, 7.0
    NB = '\u00a0'
    cols = [('#3B5998', '#EEF1F9'), ('#1F8A80', '#EAF4F2'), ('#C0882D', '#FAF3E8'), ('#B03A68', '#F9ECF1')]
    titles = ['PREPARE & APPLY', 'EVALUATE', 'AGGREGATE', 'DECIDE']
    items = [
        [('NB1  Prepare data', 'subsampling, leakage audit, stratified 80/20' + NB + 'split, train-fitted imputation and min-max scaling'),
         ('NB2  Apply methods', 'all 30 methods from four families, run on the training split, each choosing its own subset size')],
        [('NB3  Detection performance', 'AUC and MCC on the held-out test set; classifier chosen by 5-fold' + NB + 'CV; SMOTE on classifier training data only'),
         ('NB3  Stability', 'B' + NB + '=' + NB + '30 stratified bootstraps (≤' + NB + '5,000 rows), lighter selector settings (Table' + NB + '3); Nogueira and Jaccard indices'),
         ('NB3  Compactness', 'size of each selected subset')],
        [('NB4  Cross-dataset', 'average ranks, Friedman, Iman-Davenport, Nemenyi' + NB + 'CD, Kendall’s' + NB + 'W'),
         ('NB4  Families', 'four-family comparison on every axis'),
         ('NB4  Decision rule', 'stability first, compactness only as a tie-break')],
        [('Outputs', 'CD diagrams, family ranking, performance-stability trade-off, per-dataset breakdown'),
         ('Recommended method', 'the most stable method, reported with its limits and alternatives')],
    ]
    BAR_LINES = ['One protocol throughout:  fixed random seed  ·  30 methods across 4 datasets  ·  stratified 5-fold cross-validation',
                 'SMOTE on classifier training data only  ·  B = 30 bootstrap resamples']
    fig = plt.figure(figsize=(W, 3), dpi=DPI); rend = fig.canvas.get_renderer()

    def wid(s, size, weight='normal'):
        return rend.get_text_width_height_descent(s, FontProperties(family=FONT, size=size, weight=weight), ismath=False)[0] / DPI

    def wrap(s, size, maxw, weight='normal'):
        out, line = [], ''
        for wd in s.split(' '):
            t = (line + ' ' + wd).strip()
            if wid(t, size, weight) <= maxw: line = t
            else: out.append(line); line = wd
        out.append(line); return out

    pt = lambda p: p / 72.0
    LH_B, LH_H, ITEM_GAP, HEAD_GAP, PAD, BARW, IND = pt(BODY) * 1.22, pt(HEAD) * 1.28, 0.075, 0.02, 0.065, 0.025, 0.055
    mx, gap = 0.02, 0.05

    def layout(widths):
        blocks, hs = [], []
        for k, col in enumerate(items):
            inner = widths[k] - 2 * PAD - IND; bl = []
            for hd, body in col:
                hl = [hd] if wid(hd, HEAD, 'bold') <= inner else wrap(hd.replace('  ', '\u2002'), HEAD, inner, 'bold')
                bl.append((hl, wrap(body, BODY, inner)))
            blocks.append(bl)
            hs.append(sum(len(h) * LH_H + HEAD_GAP + len(l) * LH_B for h, l in bl) + ITEM_GAP * (len(bl) - 1))
        return blocks, max(hs)

    widths = [1.62, 2.10, 1.62, 1.63]          # column widths chosen to minimise the figure height
    blocks, content_h = layout(widths)
    bar_h = 2 * pt(BAR) * 1.35 + 0.08; chev_h = 0.27; box_pad = 0.085
    H = 0.02 + chev_h + 0.05 + content_h + 2 * box_pad + 0.06 + bar_h + 0.02
    fig.set_size_inches(W, H)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    y_top = H - 0.02; x = mx; box_h = content_h + 2 * box_pad
    for k in range(4):
        c, bg = cols[k]; bw = widths[k]
        ax.add_patch(Polygon([(x, y_top), (x + bw - 0.11, y_top), (x + bw, y_top - chev_h / 2), (x + bw - 0.11, y_top - chev_h), (x, y_top - chev_h)], closed=True, fc=c, ec='none'))
        ax.text(x + 0.08, y_top - chev_h / 2, str(k + 1), color='white', fontsize=TITLE + 1.5, fontweight='bold', va='center', ha='left', family=FONT)
        ax.text(x + 0.23, y_top - chev_h / 2, titles[k], color='white', fontsize=TITLE, fontweight='bold', va='center', ha='left', family=FONT)
        by = y_top - chev_h - 0.05
        ax.add_patch(FancyBboxPatch((x, by - box_h), bw, box_h, boxstyle='round,pad=0,rounding_size=0.045', fc=bg, ec='none'))
        y = by - box_pad
        for hls, lines in blocks[k]:
            ax.add_patch(plt.Rectangle((x + PAD, y - len(hls) * LH_H + 0.012), BARW, len(hls) * LH_H - 0.018, fc=c, ec='none'))
            for h in hls:
                ax.text(x + PAD + IND, y - LH_H / 2, h, fontsize=HEAD, fontweight='bold', va='center', ha='left', color='#1a1a1a', family=FONT); y -= LH_H
            y -= HEAD_GAP
            for ln in lines:
                ax.text(x + PAD + IND, y - LH_B / 2, ln, fontsize=BODY, va='center', ha='left', color='#333333', family=FONT); y -= LH_B
            y -= ITEM_GAP
        x += bw + gap
    ax.add_patch(FancyBboxPatch((mx, 0.02), W - 2 * mx, bar_h, boxstyle='round,pad=0,rounding_size=0.035', fc='#E8E8E8', ec='none'))
    for i, ln in enumerate(BAR_LINES):
        ax.text(W / 2, 0.02 + bar_h - 0.04 - pt(BAR) * 1.35 * (i + 0.5), ln, fontsize=BAR, va='center', ha='center', color='#333333', family=FONT)
    save(fig, 'Fig1_methodology', facecolor='white')


# ----------------------------------------------------------------------------- Figure 2 (critical-difference diagrams)
def cliques(ranks, cd):
    vals = sorted(ranks.values); out = []
    for i in range(len(vals)):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] < cd: j += 1
        if j > i and not any(a <= i and j <= b for a, b in out): out.append((i, j))
    return [(vals[a], vals[b]) for a, b in out]


def cd_panel(ax, ranks, family, cd, title):
    k = len(ranks); items = sorted(ranks.items(), key=lambda kv: kv[1]); half = k // 2
    ax.set_xlim(-10.5, k + 11.5); ax.set_ylim(-15.8, 6.9); ax.axis('off')
    ax.text(-10.5, 6.8, title, fontsize=9.5, fontweight='bold', va='top', ha='left')
    ax.plot([1, k], [0, 0], color='black', lw=1.2, solid_capstyle='butt')
    for t in range(1, k + 1):
        big = t % 2 == 1
        ax.plot([t, t], [0, 0.32 if big else 0.18], color='black', lw=0.9)
        if big: ax.text(t, 0.45, str(t), ha='center', va='bottom', fontsize=7.6)
    cl = sorted(cliques(ranks, cd), key=lambda ab: ab[0])
    for i, (a, b) in enumerate(cl):
        ax.plot([a - 0.1, b + 0.1], [1.75 + 0.5 * i] * 2, color='black', lw=3.6, solid_capstyle='butt')
    ycd = 1.75 + 0.5 * len(cl) + 0.75
    ax.plot([1, 1 + cd], [ycd, ycd], color='black', lw=1.0)
    for xx in (1, 1 + cd): ax.plot([xx, xx], [ycd - 0.18, ycd + 0.18], color='black', lw=1.0)
    ax.text(1 + cd / 2, ycd + 0.3, 'CD = %.2f' % cd, ha='center', va='bottom', fontsize=7.8)
    for i, (m, r) in enumerate(items[:half]):
        y = -(i + 1); ax.plot([r, r, 0.2], [0, y, y], color='#8c8c8c', lw=0.7)
        ax.text(-0.1, y, '%s (%.2f)' % (m, r), ha='right', va='center', fontsize=7.2, color=COL[family[m]])
    for i, (m, r) in enumerate(items[half:][::-1]):
        y = -(i + 1); ax.plot([r, r, k + 0.8], [0, y, y], color='#8c8c8c', lw=0.7)
        ax.text(k + 1.1, y, '(%.2f) %s' % (r, m), ha='left', va='center', fontsize=7.2, color=COL[family[m]])


def separable_pairs(ranks, cd):
    v = ranks.values
    return int(sum(abs(v[i] - v[j]) > cd for i in range(len(v)) for j in range(i + 1, len(v))))


def figure2(mats, family):
    k, n = mats['AUC'].shape; cd = nemenyi_cd(k, n); pairs = k * (k - 1) // 2
    panels = [('AUC', average_ranks(mats['AUC'], True), 'Detection performance (AUC)'),
              ('MCC', average_ranks(mats['MCC'], True), 'Detection performance (MCC)'),
              ('Nogueira', average_ranks(mats['Nogueira'], True), 'Selection stability (Nogueira index)')]
    fig, axs = plt.subplots(3, 1, figsize=(6.855, 9.785))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, hspace=0.10)
    for ax, (key, r, label), tag in zip(axs, panels, 'abc'):
        cd_panel(ax, r, family, cd, '(%s)  %s: %d of %d pairs separable' % (tag, label, separable_pairs(r, cd), pairs))
    save(fig, 'Fig2_CD_diagrams')
    print(f'  Nemenyi CD (k={k}, N={n}, alpha={ALPHA}) = {cd:.2f}')


# ----------------------------------------------------------------------------- Figure 4 (family ranks)
def figure4(mats, family):
    fams = ['Filter', 'Hybrid', 'Embedded', 'Wrapper']
    rk = pd.DataFrame({'AUC': average_ranks(mats['AUC'], True), 'MCC': average_ranks(mats['MCC'], True),
                       'Nogueira': average_ranks(mats['Nogueira'], True), 'Compactness': average_ranks(mats['Compactness'], False)})
    fm = rk.groupby(rk.index.map(family)).mean().round(2)
    fig, ax = plt.subplots(figsize=(6.14, 5.385))
    fig.subplots_adjust(left=0.125, right=0.985, top=0.9, bottom=0.13)
    x = np.arange(4); w = 0.2
    for i, f in enumerate(fams):
        xs = x + (i - 1.5) * w; vals = fm.loc[f, ['AUC', 'MCC', 'Nogueira', 'Compactness']].values
        ax.bar(xs, vals, w * 0.96, color=COL[f], label=f, edgecolor='white', linewidth=0.6)
        for xx, v in zip(xs, vals): ax.text(xx, v + 0.35, '%.1f' % v, ha='center', va='bottom', fontsize=10, color='#3a3a3a')
    ax.set_xticks(x); ax.set_xticklabels(['Performance\n(AUC)', 'Performance\n(MCC)', 'Stability\n(Nogueira)', 'Compactness'], fontsize=14)
    ax.set_ylabel('Average rank  (lower is better)', fontsize=15.5); ax.set_ylim(0, 27); ax.tick_params(axis='y', labelsize=13.5)
    ax.yaxis.grid(True, color='#d6d6d6', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=13.5, handlelength=1.4, columnspacing=1.2)
    save(fig, 'Fig4_family_ranks')
    return fm


# ----------------------------------------------------------------------------- Figure 5 (performance vs stability)
def figure5(mats, family):
    auc = mats['AUC'].mean(axis=1); nog = mats['Nogueira'].mean(axis=1).reindex(auc.index)
    rho, p = stats.spearmanr(auc, nog)
    fig, ax = plt.subplots(figsize=(5.84, 5.165))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.975, bottom=0.125)
    for f in ['Filter', 'Embedded', 'Hybrid', 'Wrapper']:
        m = [x for x in auc.index if family[x] == f]
        ax.scatter(auc[m], nog[m], s=80, color=COL[f], edgecolor='white', linewidth=0.8, label=f, zorder=3)
    LAB = {'variance': (-0.016, 0.08), 'fcbf': (-0.002, 0.075), 'boruta': (0.012, 0.065), 'rf_importance': (0.012, -0.075),
           'lasso': (-0.022, 0.06), 'rfecv': (-0.02, 0.02), 'elasticnet': (-0.024, -0.105), 'bgwo': (0.01, -0.055)}
    for m, (dx, dy) in LAB.items():
        ax.annotate(m, (auc[m], nog[m]), xytext=(auc[m] + dx, nog[m] + dy), fontsize=11, color='#3a3a3a', ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color='#8c8c8c', lw=0.7, shrinkA=3, shrinkB=3), zorder=4)
    ax.text(0.877, 0.37, 'Spearman $\\rho$ = %.3f\n(p = %.3f)' % (rho, p), fontsize=12.5, color='#3a3a3a', va='center')
    ax.set_xlim(0.875, 0.975); ax.set_ylim(-0.06, 1.09)
    ax.set_xlabel('Mean AUC  (better detection $\\rightarrow$)', fontsize=15.5)
    ax.set_ylabel('Mean Nogueira index  (more stable $\\rightarrow$)', fontsize=15.5)
    ax.tick_params(labelsize=13.5); ax.grid(True, color='#d6d6d6', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=13, handletextpad=0.3, labelspacing=0.35)
    save(fig, 'Fig5_performance_vs_stability')
    print(f'  Spearman rho (mean AUC vs mean Nogueira) = {rho:.3f}, p = {p:.3f}')



# ----------------------------------------------------------------------------- Figure 3 (mean stability per method)
def figure3(mats, family, order):
    nog = mats['Nogueira'].mean(axis=1)
    pos = {m: i for i, m in enumerate(order)}
    meths = sorted(nog.index, key=lambda m: (-round(nog[m], 12), pos[m]))   # ties keep the decision-rule order
    fig, ax = plt.subplots(figsize=(6.0, 6.53))
    fig.subplots_adjust(left=0.285, right=0.965, top=0.99, bottom=0.085)
    y = np.arange(len(meths))[::-1]
    ax.barh(y, [nog[m] for m in meths], height=0.72, color=[COL[family[m]] for m in meths])
    for yy, m in zip(y, meths):
        ax.text(nog[m] + 0.012, yy, '%.3f' % nog[m], va='center', ha='left', fontsize=10.5, color='#333333')
    ax.set_yticks(y); ax.set_yticklabels(meths, fontsize=11)
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(meths) - 0.4)
    ax.set_xlabel('Mean Nogueira stability index', fontsize=14); ax.tick_params(axis='x', labelsize=12)
    ax.xaxis.grid(True, color='#d6d6d6', lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[f]) for f in ['Filter', 'Embedded', 'Hybrid', 'Wrapper']]
    ax.legend(handles, ['Filter', 'Embedded', 'Hybrid', 'Wrapper'], loc='lower right', fontsize=12, frameon=False)
    save(fig, 'Fig3_stability_by_method')


# ----------------------------------------------------------------------------- Figure 6 (correlations between the five outcomes)
def figure6(mats):
    means = pd.DataFrame({'AUC': mats['AUC'].mean(axis=1), 'MCC': mats['MCC'].mean(axis=1), 'Nogueira': mats['Nogueira'].mean(axis=1),
                          'Jaccard': mats['Jaccard'].mean(axis=1), 'Subset size': mats['Compactness'].mean(axis=1)})
    rho = means.corr(method='spearman')
    labels = list(rho.columns); n = len(labels)
    M = np.ma.masked_array(rho.values, mask=np.triu(np.ones((n, n), bool), k=1))
    fig, ax = plt.subplots(figsize=(6.6, 5.29))
    fig.subplots_adjust(left=0.17, right=0.86, top=0.98, bottom=0.17)
    mesh = ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), M, cmap='RdBu_r', vmin=-1, vmax=1, edgecolors='white', linewidth=3)
    for i in range(n):
        for j in range(i + 1):
            v = rho.values[i, j]
            ax.text(j + 0.5, i + 0.5, '%.3f' % v, ha='center', va='center', fontsize=12, color='white' if abs(v) > 0.6 else '#222222')
    ax.set_xlim(0, n); ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n) + 0.5); ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=13)
    ax.set_yticks(np.arange(n) + 0.5); ax.set_yticklabels(labels, fontsize=13)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, ticks=np.arange(-1, 1.01, 0.25))
    cb.set_label('Spearman $\\rho$', fontsize=13); cb.ax.tick_params(labelsize=11)
    save(fig, 'Fig6_outcome_correlations')
    return rho


# ----------------------------------------------------------------------------- Figure 7 (stability rank per dataset)
def figure7(mats, family, order):
    show = {'HIPAA': 'HIPAA', 'WUSTL': 'WUSTL-EHMS', 'CICIOMT': 'CICIoMT', 'WUSTL_HDRL': 'WUSTL-HDRL'}
    ds = [d for d in show if d in mats['Nogueira'].columns]
    R = mats['Nogueira'][ds].rank(axis=0, ascending=False, method='average').loc[order]
    fig, ax = plt.subplots(figsize=(6.0, 7.18))
    fig.subplots_adjust(left=0.27, right=0.88, top=0.99, bottom=0.075)
    cmap = plt.get_cmap('YlGnBu_r'); norm = matplotlib.colors.Normalize(vmin=1, vmax=len(order))
    mesh = ax.pcolormesh(np.arange(len(ds) + 1), np.arange(len(order) + 1), R.values, cmap=cmap, norm=norm, edgecolors='white', linewidth=1.2)
    for i, m in enumerate(order):
        for j, d in enumerate(ds):
            v = R.loc[m, d]; r, g, b, _ = cmap(norm(v))
            ax.text(j + 0.5, i + 0.5, ('%d' % v) if float(v).is_integer() else ('%.1f' % v), ha='center', va='center', fontsize=9.5,
                    color='white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else '#222222')
    ax.set_xlim(0, len(ds)); ax.set_ylim(len(order), 0)
    ax.set_xticks(np.arange(len(ds)) + 0.5); ax.set_xticklabels([show[d] for d in ds], rotation=25, ha='right', fontsize=11)
    ax.set_yticks(np.arange(len(order)) + 0.5); ax.set_yticklabels(order, fontsize=10.5)
    for t, m in zip(ax.get_yticklabels(), order): t.set_color(COL[family[m]])
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.add_patch(plt.Rectangle((0, order.index('variance')), len(ds), 1, fill=False, ec='black', lw=1.6))   # recommended method
    cb = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, shrink=0.45, ticks=[5, 10, 15, 20, 25, 30])
    cb.set_label('Stability rank (1 = most stable)', fontsize=10.5); cb.ax.tick_params(labelsize=10)
    save(fig, 'Fig7_stability_rank_per_dataset')
    return R


# ----------------------------------------------------------------------------- Figure 8 (Nogueira vs Jaccard)
def figure8(mats, family):
    jac = mats['Jaccard'].mean(axis=1); nog = mats['Nogueira'].mean(axis=1).reindex(jac.index)
    rho, p = stats.spearmanr(jac, nog)
    fig, ax = plt.subplots(figsize=(5.84, 5.165))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.975, bottom=0.125)
    ax.plot([0, 1], [0, 1], ls='--', color='#888888', lw=1.1, zorder=1)
    for f in ['Filter', 'Embedded', 'Hybrid', 'Wrapper']:
        m = [x for x in jac.index if family[x] == f]
        ax.scatter(jac[m], nog[m], s=80, color=COL[f], edgecolor='white', linewidth=0.8, label=f, zorder=3)
    ax.annotate('variance', (jac['variance'], nog['variance']), xytext=(jac['variance'] - 0.06, nog['variance'] + 0.07),
                fontsize=11, color='#3a3a3a', ha='center', va='center')
    for m, (dx, dy) in {'lasso': (-0.1, 0.11), 'elasticnet': (-0.11, -0.1)}.items():
        ax.annotate(m, (jac[m], nog[m]), xytext=(jac[m] + dx, nog[m] + dy), fontsize=11, color='#3a3a3a', ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color='#8c8c8c', lw=0.9, shrinkA=3, shrinkB=4), zorder=4)
    ax.annotate('bgwo', (jac['bgwo'], nog['bgwo']), xytext=(jac['bgwo'] - 0.075, nog['bgwo'] + 0.025), fontsize=11, color='#3a3a3a',
                ha='center', va='center')
    ax.text(0.03, 0.985, 'Spearman $\\rho$ = %.3f  (p = %.4f)' % (rho, p), fontsize=12.5, color='#3a3a3a', va='top')
    ax.set_xlim(0, 1.04); ax.set_ylim(-0.05, 1.04)
    ax.set_xlabel('Mean Jaccard index (uncorrected overlap)', fontsize=15)
    ax.set_ylabel('Mean Nogueira index (chance-corrected)', fontsize=15)
    ax.tick_params(labelsize=13.5); ax.grid(True, color='#d6d6d6', lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.legend(loc='lower right', bbox_to_anchor=(1.01, 0.0), frameon=False, fontsize=12, handletextpad=0.2, labelspacing=0.3, borderaxespad=0.2)
    save(fig, 'Fig8_nogueira_vs_jaccard')
    print(f'  Spearman rho (mean Jaccard vs mean Nogueira) = {rho:.3f}, p = {p:.4f}')

# ----------------------------------------------------------------------------- helpers
def save(fig, name, **kw):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=DPI, **kw)
    plt.close(fig)
    print(f'Figure -> figures/{name}.png / .pdf')


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
    mats, fam_long = load()
    family = fam_long.map(lambda s: SHORT.get(s, s))
    table = decision_table(mats, fam_long)
    order = list(table['method'])                       # stability rank, ties broken by compactness
    figure1()
    figure2(mats, family)
    figure3(mats, family, order)
    figure4(mats, family)
    figure5(mats, family)
    figure6(mats)
    figure7(mats, family, order)
    figure8(mats, family)
