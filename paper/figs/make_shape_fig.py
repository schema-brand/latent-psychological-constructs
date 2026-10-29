"""Prototype figure: the shape of each construct's representation in the coordinate system shared by the three main LLMs
(analysis_v6/shape_v6.py). Columns, constructs (one per appraisal process); rows, LLMs; thin outlines, the six writers;
bold outline, their mean. Radius 1 (dashed circle) marks zero on each of the 16 shared dimensions."""
import json, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
HERE = Path(__file__).resolve().parent; A = HERE.parent.parent / 'analysis_v6'; OUT = HERE / 'shape_proto'; OUT.mkdir(exist_ok=True)
NAVY = '#1f4e79'; MBLUE = '#8fb0d6'; LBLUE = '#dce8f4'; GREY = '#8c8c8c'; MGREY = '#c4c4c4'; DGREY = '#4d4d4d'; INK = '#1a1a1a'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 6.5, 'savefig.dpi': 600, 'pdf.fonttype': 42})

SHOW = [('interdependent_self', 'Interdependent self'), ('perceived_fit', 'Perceived fit'), ('blame_attribution', 'Blame attribution'),
        ('favorable_evaluation', 'Favorable evaluation'), ('moral_agency', 'Moral agency'), ('warmth_toward_others', 'Warmth toward others'),
        ('goal_progress', 'Goal progress')]
z = np.load(A / 'shape_v6.npz'); U = z['unit']; cons = list(z['constructs']); models = list(z['models']); res = json.load(open(A / 'shape_v6.json'))
K = U.shape[-1]; th = np.linspace(0, 2 * np.pi, K, endpoint=False) + np.pi / 2; GAIN = 1.5

def outline(v, smooth):
    r = np.clip(1 + GAIN * v, 0.05, None)
    if not smooth:
        t = np.r_[th, th[0]]; r = np.r_[r, r[0]]; return t, r
    t = np.r_[th, th[0] + 2 * np.pi]; cs = CubicSpline(t, np.r_[r, r[0]], bc_type='periodic')
    tt = np.linspace(th[0], th[0] + 2 * np.pi, 400); return tt, np.clip(cs(tt), 0.05, None)

def figure(smooth, name):
    fig, axs = plt.subplots(len(models), len(SHOW), figsize=(7.2, 3.35), subplot_kw={'projection': 'polar'})
    fig.subplots_adjust(left=0.105, right=0.995, top=0.86, bottom=0.03, wspace=0.08, hspace=0.06)
    for i, m in enumerate(models):
        for j, (c, lab) in enumerate(SHOW):
            ax = axs[i, j]; k = cons.index(c)
            ax.set_ylim(0, 2.05); ax.set_xticks([]); ax.set_yticks([]); ax.spines['polar'].set_visible(False); ax.grid(False)
            for a in th: ax.plot([a, a], [0, 2.0], color='#eeeeee', lw=0.35, zorder=0)
            tt = np.linspace(0, 2 * np.pi, 200); ax.plot(tt, np.ones_like(tt), color=MGREY, lw=0.5, ls=(0, (2, 2)), zorder=1)
            for w in range(U.shape[2]):
                if np.any(U[k, i, w]): t, r = outline(U[k, i, w], smooth); ax.plot(t, r, color=MBLUE, lw=0.45, alpha=0.9, zorder=2)
            mv = U[k, i][np.any(U[k, i], axis=1)].mean(0); mv /= np.linalg.norm(mv)
            t, r = outline(mv, smooth); ax.fill(t, r, color=LBLUE, alpha=0.55, zorder=3, lw=0); ax.plot(t, r, color=NAVY, lw=1.0, zorder=4)
            if i == 0: ax.set_title(lab, fontsize=6.8, color=INK, pad=4)
        fig.text(0.098, 1 - (0.14 + (i + 0.5) * 0.83 / len(models)), m, ha='right', va='center', fontsize=6.5, color=DGREY)
    fig.text(0.105, 0.975, f'Representations of seven constructs in {K} dimensions shared by three LLMs', fontsize=7, color=INK, va='top')
    from matplotlib.lines import Line2D
    for xs, lw, col, lab in ((0.70, 0.45, MBLUE, 'each writer'), (0.81, 1.0, NAVY, 'mean of six writers')):
        fig.add_artist(Line2D([xs, xs + 0.025], [0.962, 0.962], color=col, lw=lw, transform=fig.transFigure))
        fig.text(xs + 0.031, 0.962, lab, va='center', fontsize=6, color=DGREY)
    fig.savefig(OUT / f'{name}.png', dpi=600, facecolor='white'); fig.savefig(OUT / f'{name}.pdf'); plt.close(fig)

figure(False, 'shape_polygon'); figure(True, 'shape_smooth')
print('identification', round(res['identification_cross_llm_cross_writer'], 3), 'same', round(res['mean_cos_same_construct_other_llm_other_writer'], 2),
      'different', round(res['mean_cos_different_constructs'], 2))
