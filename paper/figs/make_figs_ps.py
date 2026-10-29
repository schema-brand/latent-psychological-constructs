"""Figures of the latent-representation paper, Psychological Science version (22 Oct 2026). A copy of make_figs_v3.py (Nature
Communications format; style rules below unchanged) with these changes only:
  Fig. 2: 95% CIs (analysis_v6/psychsci_stats_v6.json) for the main LLMs' MTMM convergent correlations, identification (layers 30-50%
          of the depth, identification_B1), content validity (study1_content) and criterion r; panel b plots identification_B1 for the
          three main LLMs and the best-layer values (IDN) for the other LLMs.
  Fig. 4: d annotated with rho, 95% CI and BF01 (study3.per_llm_scenario); e with 95% CIs (study3.positions) and stars where the
          family-wise p within the LLM < .05 (study3.familywise); a key row replaces the direct labels of e.
  Fig. 5: c = medians with 95% CIs (analysis_v6/removal_ci_v6.json), OLMo 3.1 32B added; panel b title as a noun phrase.
  Correlation labels inside panels in APA style (no leading zero).
Fig. 3c is unchanged (see psychsci/FIG_NOTES_ps.md).

Writes figs/figs_ps/fig1.png/.pdf ... fig7.png/.pdf (width 7.08 in = 180 mm) and figs_ps/plotted_values.json (every number that
is drawn, with its source). Colour code (all figures): navy = LLM representations and their activation; greys = baselines (words,
sentence encoder, LLM before training, random directions, chance); near-black = effects on
responses where they must be distinguished from activation. Arial 6-7 pt; 8 pt bold lowercase panel letters; 7 pt panel titles.

Data: latent_2026-10-17/analysis_v6 and results_v6 (see each figure), companion release data in release_2026-10-15/agent-psychonomicus/data.
Numbers that exist only as digests outside this machine (RESULTS_INVENTORY.md sections cited) are typed in below with their source.
Fig. 7d is drawn only if analysis_v6/v3_ckpt.json exists (Study 3 over the OLMo 3 7B checkpoints).
Environment overrides for testing: FIGS_V3_OUT (output folder), FIGS_V3_CKPT (path of a v3_ckpt.json).
Run: python make_figs_ps.py [figure numbers]   (Windows: set PYTHONIOENCODING=utf-8)"""
import json, os, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch, Circle, Wedge
from matplotlib.lines import Line2D
from matplotlib.text import Text
from scipy import stats

HERE = Path(__file__).resolve().parent
LAT = HERE.parent.parent                                   # latent_2026-10-17
A = LAT / "analysis_v6"; RV = LAT / "results_v6"
REL = LAT / "external" / "agent-psychonomicus" / "data"   # REPOSITORY: was LAT.parent / "release_2026-10-15" / ...
OUT = Path(os.environ.get("FIGS_V3_OUT", HERE / "figs_ps")); OUT.mkdir(parents=True, exist_ok=True)
CKPT_JSON = Path(os.environ.get("FIGS_V3_CKPT", A / "v3_ckpt.json"))
PS = json.load(open(A / "psychsci_stats_v6.json", encoding="utf-8"))      # 95% CIs, bootstrap B = 1000 (psychsci_stats_v6.py)
RCI = json.load(open(A / "removal_ci_v6.json", encoding="utf-8"))         # removal: median share left + 95% CI per main LLM

# ---------------------------------------------------------------- style
NAVY = '#1f4e79'; BLUE = '#2f6fad'; LBLUE = '#dce8f4'; ORANGE = '#d9822b'; LORANGE = '#fbe9d7'
GREY = '#8c8c8c'; LGREY = '#eeeeee'; DGREY = '#4d4d4d'; INK = '#1a1a1a'
MGREY = '#c4c4c4'; STEM = '#d4d4d4'; MBLUE = '#8fb0d6'; BAND = '#e6e6e6'
RESP = '#262626'                                            # effects on responses (near-black)
DOT = (0, (1, 1.4))                                         # chance lines
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 6.5, 'axes.titlesize': 6.5, 'axes.labelsize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'legend.fontsize': 6, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
    'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
    'axes.labelpad': 2.5, 'axes.edgecolor': INK, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False, 'mathtext.default': 'regular',
    'legend.handlelength': 1.5, 'legend.handletextpad': 0.4, 'legend.columnspacing': 1.0, 'legend.borderaxespad': 0.0,
    'legend.borderpad': 0.0, 'legend.labelspacing': 0.3, 'axes.grid': False, 'lines.linewidth': 0.8, 'axes.unicode_minus': True})
W = 7.08                                                    # 180 mm
HMAX = 8.86                                                 # 225 mm
DZ = '$d_z$'
VALS = {}                                                   # everything plotted -> plotted_values.json


def axin(fig, left, top, width, height):
    """Axes placed in inches from the figure's top-left corner."""
    H = fig.get_figheight()
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


def panel(fig, x, y, s, title):
    """Bold panel letter and its short title; (x, y) = inches from the figure's top-left corner (top of the letter)."""
    H = fig.get_figheight()
    fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight='bold', va='top', ha='left')
    fig.text((x + 0.155) / W, 1 - (y + 0.012) / H, title, fontsize=7, va='top', ha='left', linespacing=1.15)


def save(fig, name):
    assert fig.get_figwidth() <= W + 1e-6 and fig.get_figheight() <= HMAX + 1e-6, 'figure larger than 180 x 225 mm'
    qa(fig, name)
    fig.savefig(OUT / f'{name}.png', dpi=600)
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig); print('wrote', OUT / f'{name}.png', f'({fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in)')


def qa(fig, name):
    """Report text that leaves the canvas, overlaps or touches other text, or is smaller than 6 pt."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); Wp, Hp = fig.bbox.width, fig.bbox.height; T = []

    def add(t):
        if t is None or not t.get_visible() or not t.get_text().strip(): return
        T.append((t.get_text(), t.get_window_extent(r), t.get_fontsize()))
    for t in fig.texts: add(t)
    for ax in fig.axes:
        for t in ax.texts: add(t)
        add(ax.xaxis.label); add(ax.yaxis.label); add(ax.title)
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tk in axis.get_major_ticks():
                if lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:
                    if tk.label1.get_visible(): add(tk.label1)
                    if tk.label2.get_visible(): add(tk.label2)
        if ax.get_legend():
            for t in ax.get_legend().get_texts(): add(t)
    out = []; m = 0.012 * fig.dpi                             # texts closer than ~0.9 pt count as touching
    for s, b, fs in T:
        if b.x0 < 0 or b.y0 < 0 or b.x1 > Wp or b.y1 > Hp: out.append(f'outside canvas: {s!r}')
        if fs < 6 - 1e-9: out.append(f'smaller than 6 pt ({fs}): {s!r}')
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            a, b = T[i][1], T[j][1]
            if a.x0 < b.x1 + m and b.x0 < a.x1 + m and a.y0 < b.y1 - .5 and b.y0 < a.y1 - .5:
                out.append(f'overlap or touching: {T[i][0]!r} / {T[j][0]!r}')
    print(f'[QA {name}]', 'no text overlap, nothing outside the canvas, no text below 6 pt' if not out else '')
    for s in out: print('   ', s)


def chance_text(ax, x, y, ha='left', va='center', s='chance', **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=6, color=GREY, **kw)


# ---------------------------------------------------------------- LLMs: names, order, groups (one order for every panel)
GROUPS = [("Main LLMs", ["Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B"]),
          ("Further LLMs", ["Gemma 4 26B", "Qwen3.8 27B", "Granite 4.2 30B", "gpt-oss 20B", "Gemma 4 E4B", "OLMo 3 7B"]),
          ("Qwen3.5 sizes", ["Qwen3.5 27B", "Qwen3.5 9B", "Qwen3.5 4B", "Qwen3.5 2B", "Qwen3.5 0.8B"]),
          ("Before instruction tuning", ["OLMo 3 32B (pretrained)", "Gemma 4 31B (pretrained)"])]
SHOW = {"OLMo 3 32B (pretrained)": "OLMo 3 32B", "Gemma 4 31B (pretrained)": "Gemma 4 31B"}   # shown under their group header
ALIAS = {"OLMo 3 7B Instruct": "OLMo 3 7B", "Gemma 4 26B-A4B": "Gemma 4 26B"}   # names in the v3 json files -> paper names
MAIN = ["Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B"]
WRAP = {"Qwen3.6 27B": "Qwen3.6\n27B", "Gemma 4 31B": "Gemma 4\n31B", "OLMo 3.1 32B": "OLMo 3.1\n32B", "Gemma 4 26B": "Gemma 4\n26B"}


def llm_rows(present, gap=0.5):
    """y position of every LLM (top = 0, increasing downwards), group header rows, last y."""
    y, ys, hdr = 0.0, {}, []
    for g, ms in GROUPS:
        ms = [m for m in ms if m in present]
        if not ms: continue
        if ys: y += gap
        hdr.append((y, g)); y += 1
        for m in ms: ys[m] = y; y += 1
    return ys, hdr, y - 1


def row_axis(ax, ys, hdr, ylast, labels=True):
    ticks = [y for y, _ in hdr] + list(ys.values())
    ax.set_yticks(ticks); ax.set_ylim(ylast + 0.7, -0.7)
    if labels:
        ax.set_yticklabels([g for _, g in hdr] + [SHOW.get(m, m) for m in ys])
        for k, t in enumerate(ax.get_yticklabels()):
            if k < len(hdr): t.set_fontstyle('italic'); t.set_color(DGREY)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0, pad=2)
    ax.spines['left'].set_visible(False)


# ---------------------------------------------------------------- shared data
# identification at each LLM's best middle layer (V1 digests; RESULTS_INVENTORY 1.1; as make_figs_v1.py IDN)
IDN = {"Qwen3.6 27B": .9232, "Gemma 4 31B": .9184, "OLMo 3.1 32B": .9178, "Gemma 4 26B": .9119, "Qwen3.5 27B": .9253, "Qwen3.5 9B": .9033,
       "Qwen3.5 4B": .8853, "Qwen3.5 2B": .8518, "Qwen3.5 0.8B": .8081, "Gemma 4 E4B": .8615, "OLMo 3 7B": .8879, "Qwen3.8 27B": .9185,
       "gpt-oss 20B": .8716, "Granite 4.2 30B": .899, "OLMo 3 32B (pretrained)": .9128, "Gemma 4 31B (pretrained)": .8553}
JOB = {"Qwen3.6 27B": "V610_qwen36", "Gemma 4 31B": "V611_gemma4_31b", "OLMo 3.1 32B": "V612_olmo32", "Gemma 4 26B": "V720_gemma4_26b_a4b",
       "Qwen3.5 27B": "V730_qwen35_27b", "Qwen3.5 9B": "V731_qwen35_9b", "Qwen3.5 4B": "V732_qwen35_4b", "Qwen3.5 2B": "V733_qwen35_2b",
       "Qwen3.5 0.8B": "V734_qwen35_08b", "Gemma 4 E4B": "V735_gemma4_e4b", "OLMo 3 7B": "V736_olmo7", "Qwen3.8 27B": "V740_qwen38",
       "gpt-oss 20B": "V741_gptoss20", "Granite 4.2 30B": "V742_granite42", "OLMo 3 32B (pretrained)": "V721_olmo32_base",
       "Gemma 4 31B (pretrained)": "V722_gemma4_31b_base"}
RD = pd.read_csv(A / "readout_judge_v6_all_models.csv").set_index("model").hit          # meaning identified (4 middle layers)
# multitrait-multimethod correlations, band 1 (RESULTS_INVENTORY 1.4: V1 digests dig V61x/V1 and RD:15-27):
#    same construct, different writers / different constructs, same writer / different constructs, different writers
MTMM = {"Qwen3.6 27B": (.599, .049, .044), "Gemma 4 31B": (.604, .059, .053), "OLMo 3.1 32B": (.577, .059, .052), "Gemma 4 26B": (.62, .061, .056),
        "OLMo 3 32B (pretrained)": (.58, .053, .047), "Gemma 4 31B (pretrained)": (.487, .05, .044), "Qwen3.5 27B": (.604, .05, .045),
        "Qwen3.5 9B": (.587, .052, .046), "Qwen3.5 4B": (.591, .055, .049), "Qwen3.5 2B": (.583, .054, .048), "Qwen3.5 0.8B": (.587, .056, .05),
        "Gemma 4 E4B": (.643, .066, .06), "OLMo 3 7B": (.567, .056, .05), "Qwen3.8 27B": (.601, .052, .046), "gpt-oss 20B": (.613, .063, .057),
        "Granite 4.2 30B": (.575, .056, .051)}


def r2s(v):
    """.49-style correlation text."""
    return f'{v:.2f}'.replace('0.', '.', 1).replace('-', '−')


def ci_s(ci):
    """[−.18, .33]-style text of a 95% CI of a correlation."""
    return f'[{r2s(ci[0])}, {r2s(ci[1])}]'


def hci(ax, x, y, ci, color=NAVY, z=2.5):
    """Horizontal 95% CI (whisker with short caps) of the value x in row y."""
    ax.errorbar([x], [y], xerr=[[x - ci[0]], [ci[1] - x]], fmt='none', ecolor=color, elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=z)


def vci(ax, x, y, ci, color=NAVY, z=2.5, cap=1.3):
    """Vertical 95% CI (whisker with short caps) of the value y at x."""
    ax.errorbar([x], [y], yerr=[[y - ci[0]], [ci[1] - y]], fmt='none', ecolor=color, elinewidth=0.6, capsize=cap, capthick=0.6, zorder=z)


# ================================================================ Fig. 5  the causal chain in two experiments; the nomological network
def fig1():
    H = 7.5 - 2.37
    fig = plt.figure(figsize=(W, H), dpi=600)                 # drawn at the output resolution, so measured text widths are exact
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    Y = lambda t: H - t                                       # t = inches from the top of the figure

    def T(x, t, s, **kw):
        kw.setdefault('fontsize', 6.5); kw.setdefault('zorder', 20); return ax.text(x, Y(t), s, **kw)

    def rbox(x, t, w, h, fc, ec='none', lw=0.6, r=0.035, z=2, ls='-'):   # t = top of the box
        ax.add_patch(FancyBboxPatch((x, Y(t + h)), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

    def arr(x0, t0, x1, t1, color=INK, lw=0.7, ms=6, cs=None, z=12):
        kw = dict(arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z)
        if cs: kw['connectionstyle'] = cs
        ax.add_patch(FancyArrowPatch((x0, Y(t0)), (x1, Y(t1)), **kw))

    def column(x, tc, vals, cmap, w=0.1, ch=0.065, z=12):  # column of cells centred vertically at tc
        cm = plt.get_cmap(cmap); t0 = tc - len(vals) * ch / 2
        for k, v in enumerate(vals):
            ax.add_patch(Rectangle((x, Y(t0 + (k + 1) * ch)), w, ch, fc=cm(v), ec='white', lw=0.4, zorder=z))

    def head(t, s, title):
        T(0, t, s, fontsize=8, fontweight='bold', va='top')
        T(0.155, t + 0.012, title, fontsize=7, va='top')

    def mark(x, t, kind, r=0.04, color=NAVY):
        c = (x, Y(t))
        ax.add_patch(Circle(c, r, fc=color if kind == 'held' else 'white', ec=color, lw=0.7, zorder=21))
        if kind == 'partly': ax.add_patch(Wedge(c, r, 90, 270, fc=color, ec='none', zorder=22))

    def mtext(x, t, kind, s, ha='left', **kw):
        """Result marker and its text. ha='left': marker at x, text to its right; ha='right': marker at x, text ending to its left
        (so the marker always sits next to the arrow it belongs to)."""
        kw.setdefault('fontsize', 6.3)
        mark(x, t, kind)
        if ha == 'left': return T(x + 0.075, t, s, va='center', **kw)
        return T(x - 0.075, t, s, ha='right', va='center', **kw)

    rng = np.random.default_rng(4)
    # ======================= b: one silicon sample, two experiments ===================================================
    EX = pd.read_csv(A / "example_s3_v6.csv"); EX = EX[EX.model.isin(MAIN)]
    POS = ['x_B1', 'body_B1', 'q_B1', 'dec_B1']
    tb = 0.03
    head(tb, 'a', "Activation of the mediator's representation in two experiments")
    T(1.55, tb + 0.022, '',
      va='top', fontsize=6, color=DGREY)
    BX = [(1.1, 2.38), (2.5, 3.88), (4.0, 4.92), (5.62, 6.17)]  # manipulation, scenario, question, response (x from, x to)
    XS = 5.13                                                # last word fragment before the response
    XA1, XDIV = 5.44, 5.53                                   # right end of the activation axes; divider before the response axis
    XC = [(a + b) / 2 for a, b in BX[:3]] + [XS]              # bar positions under the boxes and the last fragment
    for (a, b), s in zip(BX, ('manipulation', 'scenario description', 'outcome measure', 'response (1–9)')):
        T((a + b) / 2, tb + 0.33, s, ha='center', va='bottom', fontsize=6.5, color=DGREY)
    T(XS, tb + 0.33, 'before\nresponse', ha='center', va='bottom', fontsize=6, color=DGREY, linespacing=1.05)
    rows = [dict(exp='gray_wegner_moral_typecasting', name='Moral typecasting', med='moral agency',
                 txt=['Sam is a grown man\n(vs a baby)', 'Sam pushes a tray of glasses;\na shard cuts Roger', 'How responsible\nis Sam?'],
                 note='', ts=tb + 0.38, hs=0.40),
            dict(exp='reed_identity_salience', name='Identity salience', med='identity salience',
                 txt=['You wrote about growing\ncloser to your family\n(vs standing on your own)', 'Then you rate a\nnew product',
                      'How likely are you\nto buy it?'],
                 note='', ts=tb + 1.66, hs=0.44)]
    AYL, RYL, BH = (-4, 21), (-1.6, 10.2), 0.64                # one activation scale and one response scale for both experiments
    vb = {}
    for R in rows:
        ts, hs = R['ts'], R['hs']; q = EX[EX.exp == R['exp']]
        assert len(q) == 3, R['exp']
        st = {c: (float(q[c].mean()), float(q[c].min()), float(q[c].max())) for c in POS + ['answer_dz']}
        vb[R['name']] = {c: [round(v, 4) for v in st[c]] for c in st}
        T(0.0, ts + 0.02, R['name'], va='top', fontsize=6.5, fontweight='bold')
        T(0.0, ts + 0.14, 'mediator:\n' + R['med'], va='top', fontsize=6, color=DGREY, linespacing=1.12)
        for k, ((a, b), s) in enumerate(zip(BX[:3], R['txt'])):
            rbox(a, ts, b - a, hs, '#ececec' if k == 0 else 'white', 'none' if k == 0 else MGREY, lw=0.6, z=10)
            T((a + b) / 2, ts + hs / 2, s, ha='center', va='center', fontsize=6.3, linespacing=1.12)
        rbox(BX[3][0], ts, BX[3][1] - BX[3][0], hs, 'white', RESP, lw=0.8, z=10)
        T(sum(BX[3]) / 2, ts + hs / 2, '1 … 9', ha='center', va='center', fontsize=6.5, color=RESP)
        tm = ts + hs / 2
        arr(BX[0][1] + 0.01, tm, BX[1][0] - 0.01, tm, color=GREY); arr(BX[1][1] + 0.01, tm, BX[2][0] - 0.01, tm, color=GREY)
        arr(BX[2][1] + 0.01, tm, BX[3][0] - 0.01, tm, color=GREY)
        ax.add_patch(Circle((XS, Y(tm)), 0.03, fc=NAVY, ec='white', lw=0.5, zorder=14))
        # activation of the mediator's representation by the manipulation (mean of the three main LLMs, min-max whiskers)
        ta = ts + hs + 0.09
        axA = axin(fig, BX[0][0], ta, XA1 - BX[0][0], BH); axR = axin(fig, BX[3][0], ta, BX[3][1] - BX[3][0], BH)
        axA.set_xlim(BX[0][0], XA1); axA.set_ylim(*AYL); axR.set_xlim(*BX[3]); axR.set_ylim(*RYL)
        ax.plot([XDIV, XDIV], [Y(ta - 0.02), Y(ta + BH)], color=MGREY, lw=0.6, zorder=1)   # the response has its own axis
        axA.axhline(0, color=MGREY, lw=0.5, zorder=0); axR.axhline(0, color=MGREY, lw=0.5, zorder=0)
        for x, c in zip(XC, POS):
            m_, lo_, hi_ = st[c]; bw = 0.3 if x != XS else 0.26
            axA.bar(x, m_, width=bw, color=NAVY, lw=0, zorder=2)
            axA.plot([x, x], [lo_, hi_], color=INK, lw=0.6, zorder=3, solid_capstyle='butt')
            if m_ >= 0: axA.text(x + bw / 2 + 0.03, m_, f'{m_:.1f}', ha='left', va='center', fontsize=6, color=NAVY)
            else: axA.text(x + bw / 2 + 0.03, -0.6, f'{m_:.1f}'.replace('-', '−'), ha='left', va='top', fontsize=6, color=NAVY)   # below 0
        m_, lo_, hi_ = st['answer_dz']; xr = sum(BX[3]) / 2
        axR.bar(xr, m_, width=0.3, color=RESP, lw=0, zorder=2)
        axR.plot([xr - 0.15, xr + 0.15], [m_, m_], color=RESP, lw=1.6, zorder=3, solid_capstyle='butt')   # visible even at 0
        axR.plot([xr, xr], [lo_, hi_], color=INK, lw=0.6, zorder=3, solid_capstyle='butt')
        axR.text(xr, max(hi_, 0) + 0.45, f'{m_:.1f}' if abs(m_) >= .05 else f'{m_:.2f}'.replace('-', '−'), ha='center', va='bottom', fontsize=6,
                 color=RESP, clip_on=False)
        for a_ in (axA, axR):
            a_.set_xticks([]); a_.spines['bottom'].set_visible(False); a_.spines['top'].set_visible(False); a_.patch.set_visible(False)
        axA.set_yticks([0, 10, 20]); axA.spines['right'].set_visible(False)
        axR.set_yticks([0, 4, 8]); axR.yaxis.tick_right(); axR.spines['left'].set_visible(False); axR.spines['right'].set_visible(True)
        axA.text(XA1 - 0.05, 15.5, R['note'], ha='right', va='center', fontsize=6.5, style='italic', color=INK)
        T(0.93, ta + BH / 2, f"activation of the\nmediator's\nrepresentation ({DZ})", ha='right', va='center', fontsize=6, color=NAVY,
          linespacing=1.12)
        T(BX[3][1] + 0.2, ta + BH / 2, f'effect on the\nresponse ({DZ})', ha='left', va='center', fontsize=6, color=RESP, linespacing=1.12)

    # ======================= c: the nomological network inside the LLMs (Table 2) ======================================
    t_c = 5.46 - 2.37
    ax.plot([0, W], [Y(t_c - 0.1)] * 2, color=LGREY, lw=0.6, zorder=1)
    head(t_c, 'b', 'Nomological network in LLMs')
    tm = t_c + 1.0                                           # main row
    rbox(0.15, tm - 0.16, 1.0, 0.32, '#ececec', DGREY, lw=0.6, z=10); T(0.65, tm, 'Manipulation', ha='center', va='center')
    rbox(2.55, tm - 0.18, 1.3, 0.36, LBLUE, NAVY, lw=0.8, z=10)
    T(3.2, tm, "Mediator's representation", ha='center', va='center', color=NAVY)
    rbox(2.6, t_c + 0.27, 1.2, 0.3, 'white', DGREY, lw=0.6, z=10); T(3.2, t_c + 0.42, 'Judgment of the mediator', ha='center', va='center')
    rbox(6.15, tm - 0.16, 0.85, 0.32, 'white', RESP, lw=0.9, z=10); T(6.575, tm, 'Response', ha='center', va='center', color=RESP)
    arr(1.15, tm, 2.55, tm, color=NAVY, lw=0.8)                        # manipulation -> representation
    arr(3.2, tm - 0.18, 3.2, t_c + 0.57, color=NAVY, lw=0.8)         # representation -> judgment
    arr(3.85, tm, 6.15, tm, color=RESP, lw=0.8)                       # representation -> response
    xs_, xe_, rad = 0.65, 6.575, 0.11                                 # manipulation -> response (curved, below)
    arr(xs_, tm + 0.16, xe_, tm + 0.16, color=RESP, lw=0.7, cs=f'arc3,rad={rad}')
    tapex = tm + 0.16 + rad * (xe_ - xs_) / 2; xm = (xs_ + xe_) / 2
    xmod = 5.98
    arr(xmod, t_c + 0.38, xmod, tm - 0.03, color=DGREY, lw=0.8)     # situational moderator onto representation -> response
    arr(xm, tapex + 0.42, xm, tapex + 0.02, color=DGREY, lw=0.8)    # individual characteristics onto manipulation -> response
    C2 = {'manipulation -> representation': ('held', '42–49 of 60 experiments'),
          'representation -> judgment': ('held', '39–45 of 52 experiments'),
          'representation -> response': ('partly', ['scenario description: no relation', 'outcome measure, last fragment: ρ = .31–.45',
                                                    'removal: 39–99% of the effect left']),
          'situational moderator': ('open', 'predicted interaction in 2–5 of 7'),
          'individual characteristics': ('partly', '2 of 6 characteristics in all four LLMs')}
    mtext(1.32, tm - 0.11, 'held', C2['manipulation -> representation'][1])                     # above its arrow
    mtext(3.11, (t_c + 0.57 + tm - 0.18) / 2, 'held', C2['representation -> judgment'][1], ha='right')   # beside the vertical arrow
    k_, lines_ = C2['representation -> response']                                                # just above its arrow
    for i, s in enumerate(lines_):
        tt = tm - 0.31 + i * 0.105
        if i == 0: mtext(4.0, tt, k_, s)
        else: T(4.075, tt, s, va='center', fontsize=6.3)
    T(xmod - 0.08, t_c + 0.33, 'situational moderator', ha='right', va='center', fontsize=6.5, color=DGREY)
    mtext(xmod - 0.08, t_c + 0.46, 'open', C2['situational moderator'][1], ha='right')             # beside the moderator arrow
    T(xm + 0.08, tapex + 0.27, 'individual characteristics', ha='left', va='center', fontsize=6.5, color=DGREY)
    mtext(xm + 0.09, tapex + 0.39, 'partly', C2['individual characteristics'][1])
    for i, (kd, s) in enumerate((('held', 'shown'), ('partly', 'shown for some parts of the text or some characteristics'), ('open', 'not shown'))):   # key
        mtext(0.19, tapex + 0.06 + i * 0.12, kd, s, fontsize=6, color=DGREY)
    VALS['fig5'] = {
                    'a_example_s3_v6.csv (mean, min, max over the three main LLMs; x/body/q/dec = manipulation/scenario/question/before response, band B1)': vb,
                    'a_axis_limits': {'activation': AYL, 'response': RYL},
                    'b_Table2 (marker: held / partly / open)': C2}
    save(fig, 'fig5')


def s3_band_b1():
    """Identification at the layers 30-50% of the depth for the 16 LLMs: SUPPLEMENTARY_TABLES_v1.md, Table S3 Panel A
    (column 'Layers at 30-50% of depth', whole percentages). Returns {paper name: percent}; asserts that all 16 LLMs parse."""
    txt = open(HERE.parent / "SUPPLEMENTARY_TABLES_v1.md", encoding="utf-8").read()
    blk = txt[txt.index("**Table S3."):]; blk = blk[:blk.index("*Panel B.")]
    rows = [l for l in blk.splitlines() if l.startswith("| ") and not l.startswith("|---")]
    hdr = [h.strip() for h in rows[0].strip("|").split("|")]; col = hdr.index("Layers at 30-50% of depth")
    name = {"OLMo 3 32B base": "OLMo 3 32B (pretrained)", "Gemma 4 31B base": "Gemma 4 31B (pretrained)"}
    out = {}
    for r in rows[1:]:
        c = [x.strip() for x in r.strip("|").split("|")]; m = name.get(c[0], c[0])
        if m in IDN:
            mm = re.fullmatch(r"(\d+)%", c[col]); assert mm, (m, c[col]); out[m] = int(mm.group(1))
    assert set(out) == set(IDN), sorted(set(IDN) - set(out))
    return out


# ================================================================ Fig. 2  Study 1
def fig2():
    H = 3.32; top, hgt = 0.92, 1.95
    fig = plt.figure(figsize=(W, H))
    ys, hdr, ylast = llm_rows(IDN)
    # panels: a = multitrait-multimethod (axm), b = identification (axa), c = meaning identified (axb), d = criterion (axc)
    axm = axin(fig, 1.1, top, 1.1, hgt); axa = axin(fig, 2.34, top, 1.62, hgt); axb = axin(fig, 4.1, top, 1.08, hgt); axc = axin(fig, 6.02, top, 0.98, 1.3)
    row_axis(axm, ys, hdr, ylast); row_axis(axa, ys, hdr, ylast, labels=False); row_axis(axb, ys, hdr, ylast, labels=False)
    lev = lambda k: 1 + (0.06 + k * 0.105) / hgt              # label levels above the axes (axes fraction)
    NG = json.load(open(A / "ngram_floor_v6.json")); ENC = json.load(open(A / "encoder_floor_v6.json"))
    w1 = NG["floors"]["ng1_counts_std"]["first"]               # single words (.666)
    w3 = NG["strongest_floor"]["first"]                         # word sequences, strongest floor (.7708)
    enc = ENC["identification"]["encoder_std"]["first"]         # sentence encoder (.8018)
    ch = 1 / 66
    # a: multitrait-multimethod correlations
    # navy = same construct, different writers; grey = different constructs (mean of the same-writer and different-writer values).
    # Main LLMs: values and 95% CIs from psychsci_stats_v6.json study1_identification_mtmm (bootstrap over constructs; they differ from
    # the RESULTS_INVENTORY digests in MTMM by at most .001); other LLMs: MTMM (digests), no CI. The CIs of the two different-construct
    # correlations (each at most .014 wide) are smaller than the grey markers and are not drawn.
    S1 = PS["study1_identification_mtmm"]
    conv = {m: (S1[m]["mtmm_convergent"] if m in S1 else MTMM[m][0]) for m in ys}
    disc = {m: ((S1[m]["mtmm_hetero_same_writer"] + S1[m]["mtmm_hetero_diff_writer"]) / 2 if m in S1 else (MTMM[m][1] + MTMM[m][2]) / 2) for m in ys}
    for m, y in ys.items(): axm.plot([disc[m], conv[m]], [y, y], color=STEM, lw=0.8, zorder=1, solid_capstyle='butt')
    for m in MAIN: hci(axm, conv[m], ys[m], S1[m]["mtmm_convergent_ci"])
    axm.scatter(list(disc.values()), list(ys.values()), s=10, color=GREY, lw=0, zorder=3)
    axm.scatter(list(conv.values()), list(ys.values()), s=11, color=NAVY, lw=0, zorder=4)
    axm.set_xlim(0, .7); axm.set_xticks([0, .2, .4, .6]); axm.set_xticklabels(['0', '0.2', '0.4', '0.6'])
    axm.set_xlabel('Correlation of\nobservations')
    axm.text(0, lev(0), 'different constructs', transform=axm.transAxes, ha='left', va='bottom', fontsize=6, color=GREY)
    axm.text(1, lev(1), 'same construct,\ndifferent writers', transform=axm.transAxes, ha='right', va='bottom', fontsize=6, color=NAVY,
             linespacing=1.1)
    # b: identification (dots; reference lines labelled at their top)
    axa.axvspan(.40, .48, color=BAND, lw=0, zorder=0)
    axa.text(.44, -0.55, 'LLM before training', rotation=90, ha='center', va='top', fontsize=6, color=DGREY)
    axa.axvline(ch, color=GREY, lw=0.7, ls=DOT, zorder=1)
    chance_text(axa, ch + .015, hdr[0][0])
    tr = axa.get_xaxis_transform()
    for x, c, ls, lab, k in ((w1, GREY, (0, (3, 1.6)), f'single words {w1 * 100:.0f}%', 0), (w3, GREY, (0, (5, 1.4, 1, 1.4)), f'word sequences {w3 * 100:.0f}%', 1),
                             (enc, DGREY, '-', f'sentence encoder {enc * 100:.0f}%', 2)):
        axa.axvline(x, color=c, lw=0.7, ls=ls, zorder=1)
        axa.plot([x, x], [1, lev(k) + 0.04 / hgt], transform=tr, color=c, lw=0.5, clip_on=False, zorder=1)
        axa.text(x - .012, lev(k) + 0.04 / hgt, lab, transform=tr, ha='right', va='center', fontsize=6, color=c)
    # Which value each marker is: every LLM = identification at the layers 30-50% of the depth (band B1), one layer definition.
    # Main LLMs (Qwen3.6 27B, Gemma 4 31B, OLMo 3.1 32B): psychsci_stats_v6.json study1_identification_mtmm identification_B1 with its
    # 95% CI (bootstrap over the 66 constructs). Other 13 LLMs: SUPPLEMENTARY_TABLES_v1.md Table S3 Panel A, column "Layers at 30-50%
    # of depth" (whole percentages, plotted as proportions; no CI). The best-layer values (IDN) are no longer plotted in this panel.
    S3T = s3_band_b1()
    for m in MAIN: assert round(100 * S1[m]["identification_B1"]) == S3T[m], (m, S1[m]["identification_B1"], S3T[m])
    idn_b = {m: (S1[m]["identification_B1"] if m in S1 else S3T[m] / 100) for m in ys}
    for m in MAIN: hci(axa, idn_b[m], ys[m], S1[m]["identification_B1_ci"])
    axa.scatter([idn_b[m] for m in ys], list(ys.values()), s=11, color=NAVY, lw=0, zorder=3)
    axa.set_xlim(0, 1); axa.set_xticks([0, .2, .4, .6, .8, 1]); axa.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axa.set_xlabel('Construct identified\n(proportion of pairs)')
    # c: meaning identified from the read-out words
    rj = json.load(open(A / "readout_judge_v6.json"))
    jl = sorted([r for r in rj["by_model_source_layer"] if r["model"] == "V611_gemma4_31b" and r["source"] == "jlens"], key=lambda r: r["layer"])
    r2 = float(np.mean([r["hit"] for r in jl[1:-1]]))         # second read-out, Gemma 4 31B, without first and last layer (.356)
    axb.axvline(.10, color=GREY, lw=0.7, ls=DOT, zorder=1); chance_text(axb, .115, hdr[0][0])
    # filled = first read-out (readout_judge_v6_all_models.csv); for Qwen3.6 27B and OLMo 3.1 32B the value and 95% CI of
    # psychsci_stats_v6.json study1_content '<job>|logitlens' (bootstrap over the 66 constructs; differs from the csv by <= .001);
    # open = second read-out of Gemma 4 31B, value and CI of study1_content 'V611_gemma4_31b|jlens' (= r2 from readout_judge_v6.json).
    # Gemma 4 31B's first read-out and the other LLMs: no CI in the json.
    SC = PS["study1_content"]; CK = {"Qwen3.6 27B": "V610_qwen36|logitlens", "OLMo 3.1 32B": "V612_olmo32|logitlens"}
    hits = {m: (SC[CK[m]]["mean"] if m in CK else float(RD[JOB[m]])) for m in ys}
    g2 = SC["V611_gemma4_31b|jlens"]; assert abs(g2["mean"] - r2) < .001, (g2, r2); r2 = g2["mean"]
    for m, y in ys.items(): axb.plot([0, hits[m]], [y, y], color=STEM, lw=0.5, zorder=1, solid_capstyle='butt')
    for m, k in CK.items(): hci(axb, hits[m], ys[m], SC[k]["ci"])
    hci(axb, r2, ys["Gemma 4 31B"], g2["ci"])
    axb.scatter(list(hits.values()), list(ys.values()), s=11, color=NAVY, lw=0, zorder=3)
    axb.scatter([r2], [ys["Gemma 4 31B"]], s=11, facecolor='white', edgecolor=NAVY, lw=0.8, zorder=3)
    axb.text(g2["ci"][1] + .025, ys["Gemma 4 31B"], 'second read-out', ha='left', va='center', fontsize=6, color=NAVY)
    axb.set_xlim(0, .8); axb.set_xticks([0, .2, .4, .6, .8]); axb.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8'])
    axb.set_xlabel('Meaning identified\n(proportion of lists)')
    # d: criterion validity (horizontal: one row per main LLM, so the permutation threshold can be labelled directly)
    C = json.load(open(A / "criterion_v6.json")); x = np.arange(3)
    rr = [C[m]["criterion_r"] for m in MAIN]; nn = [C[m]["null_95"] for m in MAIN]
    CR = PS["study1_criterion"]; rci = [CR[m]["ci"] for m in MAIN]       # 95% CI, bootstrap (psychsci_stats_v6.json study1_criterion)
    for m, r_ in zip(MAIN, rr): assert abs(CR[m]["r"] - r_) < 5e-4, (m, CR[m], r_)
    for i in x:
        axc.plot([0, rr[i]], [i, i], color=STEM, lw=0.8, zorder=1, solid_capstyle='butt')   # stem light, so the CI stands out
        axc.plot([nn[i], nn[i]], [i - .24, i + .24], color=GREY, lw=1.2, zorder=3, solid_capstyle='butt')
        hci(axc, rr[i], i, rci[i], z=3.5)
    axc.scatter(rr, x, s=16, color=NAVY, lw=0, zorder=4)
    axc.text(nn[0] + .02, -.42, 'permutation,\n95th percentile', ha='left', va='bottom', fontsize=6, color=GREY, linespacing=1.1)
    axc.set_ylim(2.5, -1.25); axc.set_xlim(0, .6); axc.set_xticks([0, .2, .4, .6]); axc.set_xticklabels(['0', '0.2', '0.4', '0.6'])
    axc.set_yticks(x); axc.set_yticklabels(MAIN); axc.tick_params(axis='y', length=0, pad=2); axc.spines['left'].set_visible(False)
    axc.axvline(0, color=INK, lw=0.5, zorder=1)
    axc.set_xlabel('Representation vs\njudgment (r)')
    panel(fig, 0, 0.02, 'a', 'Convergent and discriminant validity')
    panel(fig, 2.22, 0.02, 'b', "Identification in held-out\nwriters' narratives")
    panel(fig, 3.98, 0.02, 'c', 'Content validity\n(read-out words)')
    panel(fig, 5.3, 0.02, 'd', "Criterion validity\n(the LLM's judgment)")
    VALS['fig2'] = {'a_mtmm (RESULTS_INVENTORY 1.4, band 1): same construct different writers / different constructs same writer / different both':
                        {m: list(MTMM[m]) for m in ys},
                    'a_plotted_navy_same_construct_different_writers (main LLMs: psychsci_stats_v6.json; others: MTMM)': conv,
                    'a_plotted_grey_different_constructs (mean of same-writer and different-writer values)': {m: round(v, 4) for m, v in disc.items()},
                    'a_95ci_navy_main (psychsci_stats_v6.json mtmm_convergent_ci)': {m: S1[m]["mtmm_convergent_ci"] for m in MAIN},
                    'b_identification_plotted, layers 30-50% of depth (main LLMs: psychsci_stats_v6.json identification_B1; others: SUPPLEMENTARY_TABLES_v1.md Table S3 Panel A, column Layers at 30-50% of depth, whole %)': idn_b,
                    'b_table_S3_band_B1_pct (parsed, 16 LLMs)': S3T,
                    'b_95ci_main (psychsci_stats_v6.json identification_B1_ci)': {m: S1[m]["identification_B1_ci"] for m in MAIN},
                    'b_identification_best_layer_not_plotted (IDN)': {m: IDN[m] for m in ys},
                    'c_95ci (psychsci_stats_v6.json study1_content)': {**{m: SC[k]["ci"] for m, k in CK.items()}, 'Gemma 4 31B second read-out': g2["ci"]},
                    'd_95ci (psychsci_stats_v6.json study1_criterion)': dict(zip(MAIN, rci)),
                    'b_reference': {'single words (ngram_floor_v6.json floors.ng1_counts_std.first)': w1,
                                    'word sequences (ngram_floor_v6.json strongest_floor ng3_tfidf_raw)': w3,
                                    'sentence encoder (encoder_floor_v6.json identification.encoder_std.first)': enc,
                                    'LLM before training band (OLMo 3 7B step 0, olmo7_training_v1_read.csv layer 16 .3992 to layer 0 .4795)': [.40, .48],
                                    'chance': ch},
                    'c_meaning_identified (Qwen3.6 27B, OLMo 3.1 32B: psychsci_stats_v6.json study1_content logitlens; others: readout_judge_v6_all_models.csv hit)': hits,
                    'c_second_readout_Gemma_4_31B (readout_judge_v6.json jlens layers 37/44/52)': round(r2, 3), 'c_chance': .10,
                    'd_criterion_r (criterion_v6.json)': dict(zip(MAIN, rr)), 'd_null_95': dict(zip(MAIN, nn))}
    save(fig, 'fig2')


# ================================================================ Fig. 3  Study 2
SHAPES = [('interdependent_self', 'Identity salience'), ('perceived_fit', 'Perceived fit'), ('blame_attribution', 'Blame'),
          ('favorable_evaluation', 'Favorable evaluation'), ('moral_agency', 'Moral agency'), ('warmth_toward_others', 'Interpersonal\nwarmth'),
          ('goal_progress', 'Perceived goal\nprogress')]                # one construct from each appraisal process


def shape_outline(v, th, gain=1.5):
    """Closed outline through the values of the shared dimensions (radius 1 = zero), periodic cubic interpolation."""
    from scipy.interpolate import CubicSpline
    r = np.clip(1 + gain * v, 0.05, None); t = np.r_[th, th[0] + 2 * np.pi]
    tt = np.linspace(th[0], th[0] + 2 * np.pi, 400)
    return tt, np.clip(CubicSpline(t, np.r_[r, r[0]], bc_type='periodic')(tt), 0.05, None)


def shapes_panel(fig, y0):
    """Panel a: profiles of seven constructs in the dimensions shared by the three main LLMs (analysis_v6/shape_v6.py)."""
    z = np.load(A / "shape_v6.npz"); U = z['unit']; cons = list(z['constructs']); models = list(z['models'])
    res = json.load(open(A / "shape_v6.json")); K = U.shape[-1]; th = np.linspace(0, 2 * np.pi, K, endpoint=False) + np.pi / 2
    x0, cw, ch, s = 0.80, (W - 0.82) / len(SHAPES), 0.84, 0.80
    for i, m in enumerate(models):
        yc = y0 + 0.44 + i * ch
        fig.text(0.74 / W, 1 - (yc + ch / 2) / fig.get_figheight(), m, ha='right', va='center', fontsize=6.5, color=DGREY)
        for j, (c, lab) in enumerate(SHAPES):
            H = fig.get_figheight(); k = cons.index(c)
            ax = fig.add_axes([(x0 + j * cw + (cw - s) / 2) / W, 1 - (yc + (ch + s) / 2) / H, s / W, s / H], projection='polar')
            ax.set_ylim(0, 2.0); ax.set_xticks([]); ax.set_yticks([]); ax.spines['polar'].set_visible(False); ax.grid(False)
            for a in th: ax.plot([a, a], [0, 1.95], color=LGREY, lw=0.35, zorder=0)
            tt = np.linspace(0, 2 * np.pi, 200); ax.plot(tt, np.ones_like(tt), color=MGREY, lw=0.5, ls=(0, (2, 2)), zorder=1)
            for w in range(U.shape[2]):
                if np.any(U[k, i, w]): ax.plot(*shape_outline(U[k, i, w], th), color=MBLUE, lw=0.45, zorder=2)
            mv = U[k, i][np.any(U[k, i], axis=1)].mean(0); mv /= np.linalg.norm(mv); t, r = shape_outline(mv, th)
            ax.fill(t, r, color=LBLUE, alpha=0.6, lw=0, zorder=3); ax.plot(t, r, color=NAVY, lw=1.0, zorder=4)
            if i == 0: fig.text((x0 + (j + .5) * cw) / W, 1 - (y0 + 0.33) / H, lab, ha='center', va='bottom', fontsize=6.5)
    H = fig.get_figheight(); ky = 1 - (y0 + 0.065) / H                    # on the line of the panel title, clear of two-line labels
    for xs, lw, col, lab in ((5.02, 0.45, MBLUE, 'each writer'), (5.82, 1.0, NAVY, 'mean of six writers')):
        fig.add_artist(Line2D([xs / W, (xs + 0.18) / W], [ky, ky], color=col, lw=lw, transform=fig.transFigure))
        fig.text((xs + 0.23) / W, ky, lab, va='center', fontsize=6, color=DGREY)
    VALS['fig3_a'] = {'source': 'analysis_v6/shape_v6.npz, shape_v6.json', 'constructs': [c for c, _ in SHAPES], 'dimensions': K,
                      'identification_one_writer_one_llm_vs_other_llms_other_writers': res['identification_cross_llm_cross_writer'],
                      'by_llm': res['by_llm'], 'mean_cos_same_construct': res['mean_cos_same_construct_other_llm_other_writer'],
                      'mean_cos_different_constructs': res['mean_cos_different_constructs']}


def fig3():
    HA = 3.05; H = 2.46 + HA; top, h = 0.48 + HA, 1.55
    fig = plt.figure(figsize=(W, H))
    shapes_panel(fig, 0.0)
    panel(fig, 0, 0.02, 'a', 'Representations of seven constructs in 16 dimensions shared by three LLMs')
    G = np.load(A / "construct_geometry_v6.npy"); IX = json.load(open(A / "construct_geometry_v6_index.json"))
    iu = np.triu_indices(G.shape[1], 1); q = G[IX["order"].index("Qwen3.6 27B")][iu]; g = G[IX["order"].index("Gemma 4 31B")][iu]
    rho = stats.spearmanr(q, g)[0]
    axa = axin(fig, 0.48, top, 1.9, h)
    axa.scatter(q, g, s=1.6, color=NAVY, alpha=.45, lw=0, rasterized=True)
    axa.set_xlabel('Similarity, Qwen3.6 27B'); axa.set_ylabel('Similarity, Gemma 4 31B')
    axa.text(.04, .97, f'ρ = {r2s(rho)}', transform=axa.transAxes, va='top', fontsize=6.5)
    lim = (-.3, .65); tk = [-.25, 0, .25, .5]; tl = ['−0.25', '0', '0.25', '0.5']
    axa.set_xlim(*lim); axa.set_ylim(*lim); axa.set_xticks(tk); axa.set_yticks(tk); axa.set_xticklabels(tl); axa.set_yticklabels(tl)
    # b: same vs different process (labels written inside the bars of OLMo 3.1 32B)
    S = json.load(open(A / "structure_v6.json"))["theory_structure"]; E = json.load(open(A / "encoder_floor_v6.json"))["theory_structure"]["encoder"]
    sysn = [("Words", S["word counts"], GREY), ("Sentence\nencoder", E, GREY)] + [(WRAP[m], S[m], NAVY) for m in MAIN]
    axb = axin(fig, 3.45, top, 3.5, h); x = np.arange(len(sysn)); bw = .36
    for i, (lab, d, c) in enumerate(sysn):
        axb.bar(i - bw / 2, d["within"], width=bw, color=c, lw=0)
        axb.bar(i + bw / 2, d["between"], width=bw, color='white', edgecolor=c, lw=0.7)
    il = len(sysn) - 1
    axb.text(il - bw / 2, .0025, 'same process', rotation=90, ha='center', va='bottom', fontsize=6, color='white')
    axb.text(il + bw / 2, .0025, 'different processes', rotation=90, ha='center', va='bottom', fontsize=6, color=NAVY)
    axb.set_xticks(x); axb.set_xticklabels([s[0] for s in sysn]); axb.tick_params(axis='x', length=0, pad=2.5)
    axb.set_ylim(0, .1); axb.set_yticks([0, .025, .05, .075, .1]); axb.set_yticklabels(['0', '0.025', '0.050', '0.075', '0.100'])
    axb.set_ylabel('Mean similarity of constructs')
    panel(fig, 0, 0.02 + HA, 'b', 'Similarity of the 66 constructs\nin two LLMs')
    panel(fig, 2.95, 0.02 + HA, 'c', 'Similarity within and between processes')
    VALS['fig3'] = {'a_n_pairs': int(len(q)), 'a_spearman_rho (construct_geometry_v6.npy)': round(float(rho), 4),
                    'b_within_between (structure_v6.json; encoder from encoder_floor_v6.json)': {s[0].replace('\n', ' '): [s[1]["within"], s[1]["between"]] for s in sysn}}
    save(fig, 'fig3')


# ================================================================ Fig. 4  Study 3
def fig4():
    H = 7.96
    fig = plt.figure(figsize=(W, H))
    S3 = PS["study3"]
    V3 = {ALIAS.get(k, k): v for k, v in json.load(open(A / "v3_all_models.json")).items()}
    SR = {ALIAS.get(k, k): v for k, v in json.load(open(A / "v3_shift_by_repl.json")).items()}
    SH = pd.read_csv(A / "v3_shift_by_repl.csv"); SH["model"] = SH.model.replace(ALIAS)
    num = lambda s: int(str(s).split("/")[0])
    unknown = [m for m in V3 if not any(m in ms for _, ms in GROUPS)]
    if unknown: print('  !! Fig. 4: LLMs without a group (not drawn):', unknown)
    ys, hdr, ylast = llm_rows(V3)
    t1, h1, t2, h2 = 0.3, 1.95, 3.28, 1.95
    # a: experiments with greater activation of the mediator's representation
    axa = axin(fig, 1.0, t1, 2.45, h1); row_axis(axa, ys, hdr, ylast)
    own = {m: num(V3[m]["own_shift_pos"]) for m in ys}
    axa.axvline(30, color=GREY, lw=0.7, ls=DOT, zorder=1); chance_text(axa, 31, hdr[0][0])
    for m, y in ys.items(): axa.plot([0, own[m]], [y, y], color=STEM, lw=0.5, zorder=1, solid_capstyle='butt')
    axa.scatter(list(own.values()), list(ys.values()), s=11, color=NAVY, lw=0, zorder=3)
    axa.set_xlim(0, 60); axa.set_xticks([0, 10, 20, 30, 40, 50, 60])
    axa.set_xlabel("Experiments with greater activation of\nthe mediator's representation (of 60)")
    # c: activation during the scenario vs effect on the response, three main LLMs
    cw, cg, cl = 1.55, 0.3, 1.35
    allq = SH[SH.model.isin(MAIN)].dropna(subset=["dz", "shift_dz"])
    XMAX = 4.5                                                  # effects beyond this are drawn at the edge (triangles)
    xl = (-1.5, XMAX + .15); yl_ = (np.floor(allq.shift_dz.min()) - .5, np.ceil(allq.shift_dz.max()) + .5)
    rho_c, clipped = {}, {}
    for k, m in enumerate(MAIN):
        ax = axin(fig, cl + k * (cw + cg), t2, cw, h2); q = allq[allq.model == m]
        rho_c[m] = stats.spearmanr(q.dz, q.shift_dz)[0]
        ax.axhline(0, color=MGREY, lw=0.5, zorder=0); ax.axvline(0, color=MGREY, lw=0.5, zorder=0)
        inn, out = q[q.dz <= XMAX], q[q.dz > XMAX]; clipped[m] = out[["exp", "dz", "shift_dz"]].round(3).values.tolist()
        ax.scatter(inn.dz, inn.shift_dz, s=6, color=NAVY, lw=0, alpha=.85, zorder=3)
        ax.scatter([XMAX + .1] * len(out), out.shift_dz, s=9, marker='>', color=NAVY, lw=0, zorder=3, clip_on=False)
        ax.set_xlim(*xl); ax.set_ylim(*yl_); ax.set_xticks([0, 2, 4]); ax.set_yticks([-5, 0, 5, 10]); ax.set_title(m, fontsize=6.5, pad=3)
        # rho (activation at the scenario, layers 30-50%), 95% CI and BF01: psychsci_stats_v6.json study3.per_llm_scenario
        P = S3["per_llm_scenario"][m]; assert abs(P["rho"] - rho_c[m]) < 5e-4 and P["n"] == len(q), (m, P, rho_c[m], len(q))
        ax.text(.95, .97, f'ρ = {r2s(P["rho"])}\n{ci_s(P["ci"])}\nBF$_{{01}}$ = {P["BF01"]:.2f}', transform=ax.transAxes, va='top',
                ha='right', fontsize=6, linespacing=1.15)
        if k == 0: ax.set_ylabel(f"Activation of the mediator's\nrepresentation, scenario\ndescription ({DZ})")
        else: ax.set_yticklabels([])
        if k == 1: ax.set_xlabel(f'Effect on the response ({DZ})', color=RESP)
    # b: replication outcome, activation vs response (same rows as a)
    dl, dw, dg = 4.3, 1.2, 0.2
    ax1 = axin(fig, dl, t1, dw, h1); ax2 = axin(fig, dl + dw + dg, t1, dw, h1)
    row_axis(ax1, ys, hdr, ylast, labels=False); row_axis(ax2, ys, hdr, ylast, labels=False)
    rep = {m: (num(SR[m]["shift_pos_repl"]), num(SR[m]["shift_pos_fail"])) for m in ys}
    ans = {m: (num(V3[m]["repl_sig"]), num(V3[m]["fail_sig"])) for m in ys}
    for ax, D, ttl, note, c in ((ax1, rep, 'Activation', '', NAVY), (ax2, ans, 'Response', '', RESP)):
        for m, y in ys.items():
            r_, f_ = D[m]
            ax.plot([r_, f_], [y, y], color=MGREY, lw=1.0, zorder=1, solid_capstyle='butt')
            ax.scatter([f_], [y], s=13, facecolor='white', edgecolor=c, lw=0.8, zorder=2)
            ax.scatter([r_], [y], s=13, color=c, lw=0, zorder=3)
        ax.set_xlim(0, 14.5); ax.set_xticks([0, 7, 14]); ax.set_title(ttl, fontsize=6.5, pad=3, color=c)
        ax.text(.5, 1.0 + 0.03 / h2, note, transform=ax.transAxes, ha='center', va='bottom', fontsize=6, style='italic', color=DGREY)
    r_, f_ = ans[MAIN[0]]                                       # direct labels on the first row of the response half
    ax2.text(f_, hdr[0][0], 'failed', ha='center', va='center', fontsize=6, color=RESP)
    ax2.text(r_, hdr[0][0], 'replicated', ha='center', va='center', fontsize=6, color=RESP)
    fig.text((dl + dw + dg / 2) / W, 1 - (t1 + h1 + 0.135) / H, 'Experiments (of 14)', ha='center', va='top', fontsize=6.5)
    # d: correlation between activation and the effect on the response, by position and depth (depth_s3_spec_v6.json)
    SP = json.load(open(A / "depth_s3_spec_v6.json"))
    t3, h3, ew, eg, el = t2 + h2 + 0.98, 1.35, 1.78, 0.42, 0.62
    POS = [("x", "manipulation"), ("body", "scenario\ndescription"), ("q", "outcome\nmeasure"), ("dec", "before\nresponse")]
    vals_e = {}; axe = []
    for k, m in enumerate(MAIN):
        ax = axin(fig, el + k * (ew + eg), t3, ew, h3); vals_e[m] = {}; axe.append(ax)
        ax.axvspan(1.5, 3.55, color='#eaf1f8', lw=0, zorder=0)
        ax.axhline(0, color=MGREY, lw=0.5, zorder=1)
        for i, (pk, _) in enumerate(POS):
            for j, (g, off, mfc) in enumerate((("B1", -.13, NAVY), ("B2", .13, 'white'))):
                r = SP[m][f"{pk}_{g}"]; vals_e[m][f"{pk}_{g}"] = r
                ax.plot([i + off - .1, i + off + .1], [r["null_95"]] * 2, color=GREY, lw=1.0, zorder=2, solid_capstyle='butt')
                # 95% CI (psychsci_stats_v6.json study3.positions) and star if p_fwe_within_llm < .05 (study3.familywise)
                pc = S3["positions"][m][f"{pk}_{g}"]; fw = S3["familywise"][m][f"{pk}_{g}"]
                assert abs(pc["rho"] - r["rho_own"]) < 5e-4 and abs(fw["rho"] - r["rho_own"]) < 5e-4, (m, pk, g)
                vci(ax, i + off, r["rho_own"], pc["ci"], cap=1.0)
                ax.scatter([i + off], [r["rho_own"]], s=13, facecolor=mfc, edgecolor=NAVY, lw=0.8, zorder=3)
                if fw["p_fwe_within_llm"] < .05:
                    ax.scatter([i + off], [pc["ci"][1] + .055], s=16, marker='*', color=INK, lw=0, zorder=3)
                vals_e[m][f"{pk}_{g}"] = {**r, 'ci95': pc["ci"], 'p_fwe_within_llm': fw["p_fwe_within_llm"],
                                          'star': fw["p_fwe_within_llm"] < .05}
        ax.set_xlim(-.55, 3.55); ax.set_ylim(-.34, .74); ax.set_yticks([-.2, 0, .2, .4, .6])
        ax.set_yticklabels(['−0.2', '0', '0.2', '0.4', '0.6'])
        ax.set_xticks(range(4)); ax.set_xticklabels([lab for _, lab in POS]); ax.tick_params(axis='x', length=0, pad=2.5)
        ax.set_title(m, fontsize=6.5, pad=3)
        if k == 0: ax.set_ylabel('Activation vs response,\nacross experiments (ρ)')
        else: ax.set_yticklabels([])
    # key row under the panel title (the CIs leave no room for direct labels inside the small panels)
    kh = [Line2D([], [], ls='none', marker='o', ms=3.6, mfc=NAVY, mec=NAVY, mew=0.8),
          Line2D([], [], ls='none', marker='o', ms=3.6, mfc='white', mec=NAVY, mew=0.8),
          Line2D([], [], color=GREY, lw=1.0, solid_capstyle='butt'),
          Line2D([], [], ls='none', marker='*', ms=4.6, mfc=INK, mec=INK, mew=0)]
    fig.legend(kh, ['layers 30–50% of depth', 'layers 60–75% of depth', "other experiments' mediators, 95th percentile",
                    'family-wise p < .05 within the LLM'], ncol=4, loc='center left', bbox_to_anchor=(el / W, 1 - (t3 - 0.29) / H),
               handlelength=1.0, columnspacing=1.6, fontsize=6)
    panel(fig, 0, 0.02, 'a', "Activation of the mediator's representation")
    panel(fig, 3.85, 0.02, 'b', 'Replicated vs failed effects')
    panel(fig, 0, t2 - 0.36, 'c', 'Activation (scenario description) vs effect on the response')
    panel(fig, 0, t3 - 0.52, 'd', 'Activation–response correlation by position and depth')
    VALS['fig4'] = {'a_own_shift_pos_of_60 (v3_all_models.json)': own, 'a_chance': 30,
                    'c_rho_shift_vs_answer_dz (v3_shift_by_repl.csv)': {m: round(float(v), 4) for m, v in rho_c.items()},
                    'c_axis_limits': {'x': xl, 'y': yl_}, 'c_answers_beyond_axis_drawn_as_triangles [exp, dz, shift_dz]': clipped,
                    'b_representation_pos_replicated_vs_not_of_14 (v3_shift_by_repl.json)': rep,
                    'b_answer_sig_replicated_vs_not_of_14 (v3_all_models.json)': ans,
                    'd_rho_by_position_and_depth (depth_s3_spec_v6.json: rho_own, null_95, p_perm; psychsci_stats_v6.json: ci95, p_fwe_within_llm)': vals_e,
                    'c_annotation (psychsci_stats_v6.json study3.per_llm_scenario)': {m: S3['per_llm_scenario'][m] for m in MAIN},
                    'd_shaded_positions': ['outcome measure', 'before response']}
    save(fig, 'fig4')


# ================================================================ Fig. 5  Study 4
def dig(job, step, col):
    rows = [l for l in open(RV / job / step / "digest.md", encoding="utf-8").read().splitlines() if l.startswith("| ") and not l.startswith("|---")]
    hdr = [h.strip() for h in rows[0].strip("|").split("|")]
    D = pd.DataFrame([[c.strip() for c in r.strip("|").split("|")] for r in rows[1:]], columns=hdr)
    return pd.to_numeric(D.set_index(hdr[0])[col], errors="coerce")


def dig_count(job, step, pattern):
    m = re.search(pattern, open(RV / job / step / "digest.md", encoding="utf-8").read())
    return int(m.group(1)) if m else None


def fig5():
    H = 2.46; top, h = 0.54, 1.5
    fig = plt.figure(figsize=(W, H))
    # a: inducing a construct's representation, effect on that judgment minus the other constructs' representations (V4a digests)
    axa = axin(fig, 0.5, top, 1.42, h); va, npos = {}, {}
    for k, m in enumerate(MAIN):
        v = dig(JOB[m], "V4a_causal", "own - others d_z").dropna().values; va[m] = v
        npos[m] = dig_count(JOB[m], "V4a_causal", r"own - others > 0: (\d+)/66")   # digest header = the count reported in the text
        xs = k + (np.random.default_rng(k).random(len(v)) - .5) * .5
        axa.scatter(xs, v, s=4, color=NAVY, alpha=.75, lw=0, zorder=3)
        axa.plot([k - .3, k + .3], [np.median(v)] * 2, color=INK, lw=1.2, zorder=4, solid_capstyle='butt')
    axa.axhline(0, color=GREY, lw=0.5, zorder=0)
    lo_, hi_ = min(v.min() for v in va.values()), max(v.max() for v in va.values())
    axa.set_ylim(np.floor(lo_ * 2) / 2 - .1, np.ceil(hi_ * 2) / 2 + .55)
    axa.set_yticks(np.arange(np.ceil(axa.get_ylim()[0]), np.floor(axa.get_ylim()[1]) + 1))
    for k, m in enumerate(MAIN):
        axa.text(k, axa.get_ylim()[1], f'{npos[m]}/{len(va[m])}', ha='center', va='top', fontsize=6)
    axa.set_xlim(-.55, 2.55); axa.set_xticks(range(3)); axa.set_xticklabels([WRAP[m] for m in MAIN]); axa.tick_params(axis='x', length=0, pad=2.5)
    axa.set_ylabel(f'Effect on the judgment ({DZ}),\nown minus other constructs')
    # b: judgment vs response under the same inductions (counts of 52; RESULTS_INVENTORY 4.2-4.3: V4b, V6b, V5a digests and RD lines)
    L4 = ["Qwen3.6 27B", "Gemma 4 31B", "OLMo 3.1 32B", "Gemma 4 26B"]
    ans_id = [32, 32, 30, 25]; ans_d50 = [36, 34, 30, 36]; ans_d70 = [41, 36, 33, 38]; jud_id = [40, 39, 45, 40]
    chk = {"V4b own push (identification layer)": [(dig_count(JOB[m], "V4b_steer", r"own push > 0: (\d+)/52"), ans_id[i]) for i, m in enumerate(MAIN)],
           "V5a judgment (identification layer)": [(dig_count(j, "V5a_incontext", r"own push on the judgment > 0: (\d+)/52"), jud_id[i])
                                                    for i, j in enumerate(["V710_follow_qwen36", "V711_follow_gemma4_31b"])]}
    t6b = open(RV / "V810_steer2_qwen36" / "V6b_steer2" / "digest.md", encoding="utf-8").read()
    chk["V6b Qwen3.6 half depth / 70% depth"] = [(int(re.search(r"\| d50\|sit \| (\d+)/52", t6b).group(1)), ans_d50[0]),
                                                  (int(re.search(r"\| d70\|sit2end \| (\d+)/52", t6b).group(1)), ans_d70[0])]
    for k_, pairs in chk.items():
        for got, typed in pairs:
            if got is not None and got != typed: print(f'  !! Fig. 5b: {k_}: digest {got} vs typed {typed}')
    # grouped by what was measured (judgment, then the response at three depths); within each group the four LLMs in a fixed order
    axb = axin(fig, 2.58, top, 2.32, h); bw = .2; GX = [0, 1.42, 2.42, 3.42]
    CONDS = [('judgment,\nearly layer', jud_id, NAVY), ('response,\nearly layer', ans_id, RESP), ('response,\nhalf depth', ans_d50, RESP),
             ('response,\n70% depth', ans_d70, RESP)]
    for g, (lab, vals, c) in zip(GX, CONDS):
        for i, v in enumerate(vals):
            xx = g + (i - 1.5) * (bw + .025)
            axb.bar(xx, v, width=bw, color=c, lw=0, zorder=2)
            axb.text(xx, v + .7, str(v), ha='center', va='bottom', fontsize=6, color=c)
    for i, m in enumerate(L4):                                  # the LLMs, written once inside the judgment bars
        axb.text((i - 1.5) * (bw + .025), 1.2, m, rotation=90, ha='center', va='bottom', fontsize=6, color='white')
    axb.axhline(26, color=GREY, lw=0.7, ls=DOT, zorder=3); chance_text(axb, (GX[0] + GX[1]) / 2, 26.6, ha='center', va='bottom')
    axb.set_xlim(-.5, 3.92); axb.set_ylim(0, 52); axb.set_yticks([0, 13, 26, 39, 52])
    axb.set_xticks(GX); axb.set_xticklabels([c[0] for c in CONDS]); axb.tick_params(axis='x', length=0, pad=2.5)
    for t, (_, _, c) in zip(axb.get_xticklabels(), CONDS): t.set_color(c)
    axb.set_ylabel('Experiments with the predicted\neffect (of 52)')
    # c: removal (V4c mean replacement): median share of the manipulation's effect on the response left, with 95% CI, per main LLM
    # (analysis_v6/removal_ci_v6.json). Mapping of the categories (as in make_figs_v3.py, whose typed values for Qwen3.6 27B and
    # Gemma 4 31B are the rounded medians of these keys): mediator = own_spec_k10__mean, similar constructs = near_spec_k10__mean
    # (the constructs most similar to the mediator), other constructs = rand_spec_k10__mean, random directions = unif__mean.
    # OLMo 3.1 32B added. Bars = medians, whiskers = 95% CIs; numbers above the bars dropped (no room next to the CIs).
    L3 = MAIN
    KEYS = {"mediator": "own_spec_k10__mean", "similar constructs": "near_spec_k10__mean", "other constructs": "rand_spec_k10__mean",
            "random directions": "unif__mean"}
    rem = {lab: [100 * RCI[m][k]["median"] for m in L3] for lab, k in KEYS.items()}
    remci = {lab: [[100 * v for v in RCI[m][k]["ci"]] for m in L3] for lab, k in KEYS.items()}
    typed_v3 = {"mediator": [68, 39], "similar constructs": [86, 17], "other constructs": [93, 83], "random directions": [98, 84]}
    for lab, v in typed_v3.items():                             # same numbers as the NC version for the two LLMs it showed
        assert [int(round(x)) for x in rem[lab][:2]] == v, (lab, rem[lab][:2], v)
    sty = {"mediator": dict(facecolor=NAVY, edgecolor=NAVY, lw=0.7), "similar constructs": dict(facecolor=MBLUE, edgecolor=MBLUE, lw=0.7),
           "other constructs": dict(facecolor='#a6a6a6', edgecolor='#a6a6a6', lw=0.7), "random directions": dict(facecolor=LGREY, edgecolor=GREY, lw=0.7)}
    tcol = {"mediator": 'white', "similar constructs": INK, "other constructs": INK, "random directions": INK}
    axc = axin(fig, 5.5, top, 1.5, h); x = np.arange(len(L3)); bw = .2
    for k, (lab, vals) in enumerate(rem.items()):
        off = (k - 1.5) * (bw + .015)
        axc.bar(x + off, vals, width=bw, zorder=2, **sty[lab])
        for xi, v, ci in zip(x + off, vals, remci[lab]): vci(axc, xi, v, ci, color=INK, z=3, cap=1.0)
        axc.text(off, 3, lab, rotation=90, ha='center', va='bottom', fontsize=6, color=tcol[lab], zorder=4)   # labels inside the first group
    axc.axhline(100, color=GREY, lw=0.7, ls=DOT, zorder=1)
    axc.set_xlim(-.55, len(L3) - .45); axc.set_ylim(0, 112); axc.set_yticks([0, 25, 50, 75, 100])
    axc.set_xticks(x); axc.set_xticklabels([WRAP[m] for m in L3]); axc.tick_params(axis='x', length=0, pad=2.5)
    axc.set_ylabel("Manipulation's effect on the\nresponse left after removal (%)")
    panel(fig, 0, 0.02, 'a', 'Induced representations\nand judgments')
    panel(fig, 2.02, 0.02, 'b', 'Induction effects on judgment and response')
    panel(fig, 4.98, 0.02, 'c', "Removal of the mediator's\nrepresentation")
    VALS['fig6'] = {'a_own_minus_others_dz (V4a digests)': {m: {'n': len(v), 'n_pos_digest_header (shown)': npos[m], 'n_pos_table_values': int((v > 0).sum()),
                                                                 'median': round(float(np.median(v)), 3)} for m, v in va.items()},
                    'b_counts_of_52': {'LLMs': L4, 'answer_identification_layer': ans_id, 'answer_half_depth': ans_d50, 'answer_70pct_depth': ans_d70,
                                       'judgment_identification_layer': jud_id, 'chance': 26},
                    'c_share_left_pct_median (removal_ci_v6.json)': {'LLMs': L3, 'mapping': KEYS, **rem},
                    'c_95ci_pct (removal_ci_v6.json)': remci,
                    'c_n_experiments': {m: {'total': RCI[m]['n_experiments'], **{lab: RCI[m][k]['n'] for lab, k in KEYS.items()}} for m in L3}}
    save(fig, 'fig6')


# ================================================================ Fig. 6  Study 5
def fig6():
    H = 2.86; top, h = 0.66, 1.78
    fig = plt.figure(figsize=(W, H))
    TESTS = ["thaler_costs_justify", "freedman_consistency", "mogilner_materialism", "whiteargo_cse", "kivetz_guilt", "reed_importance", "thaler_selfmonitoring_null"]
    TL = ["Beer price × costs justify prices", "Foot in the door × consistency", "Time vs money × materialism", "Identity threat × collective self-esteem",
          "Effort × luxury guilt", "Identity salience × family importance", "Beer price × self-monitoring (null)"]
    # interaction d_z with the characteristic described in the agent profile (version A; V7i digests via DEVLOG 607 = RESULTS_INVENTORY 5)
    VA = {"Qwen3.6 27B": [4.302, 1.467, 0.406, 0.876, -0.078, -0.913, 0.408], "Gemma 4 31B": [2.609, 1.148, -0.2, -0.288, 0.272, -1.001, 0.213],
          "OLMo 3.1 32B": [2.077, 1.32, 0.719, -0.343, -0.565, 0.903, 0.357], "Gemma 4 26B": [2.333, 1.15, 0.087, -0.142, 0.217, -0.613, 0.57]}
    n_agents = 18; tcrit = stats.t.ppf(.975, n_agents - 1)      # t = d_z * sqrt(n) reproduces the t values in DEVLOG 607
    ylab = {t: i for i, t in enumerate(TESTS)}; ylab[TESTS[-1]] += .45
    axa = axin(fig, 1.68, top, 2.0, h)
    MK = {"Qwen3.6 27B": 'o', "Gemma 4 31B": 's', "OLMo 3.1 32B": '^', "Gemma 4 26B": 'D'}
    offs = np.linspace(-.27, .27, 4)
    axa.axvline(0, color=GREY, lw=0.5, zorder=0)
    sig = {}
    for k, (m, v) in enumerate(VA.items()):
        for t, d in zip(TESTS, v):
            s_ = abs(d) * np.sqrt(n_agents) > tcrit; sig[(m, t)] = bool(s_)
            axa.scatter(d, ylab[t] + offs[k], s=9 if MK[m] != 'D' else 7, marker=MK[m], facecolor=NAVY if s_ else 'white', edgecolor=NAVY, lw=0.6, zorder=3)
    axa.set_yticks([ylab[t] for t in TESTS]); axa.set_yticklabels(TL); axa.set_ylim(max(ylab.values()) + .6, -.6)
    axa.tick_params(axis='y', length=0, pad=2); axa.spines['left'].set_visible(False)
    axa.set_xlim(-1.5, 4.7); axa.set_xticks([-1, 0, 1, 2, 3, 4])
    axa.set_xlabel(f'Interaction with the characteristic\ndescribed in the agent profile ({DZ})')
    hd = [Line2D([], [], ls='none', marker=MK[m], ms=3.2 if MK[m] != 'D' else 2.8, mfc=NAVY, mec=NAVY) for m in VA]
    axa.legend(hd, list(VA), ncol=4, loc='lower left', bbox_to_anchor=(-1.6 / 2.0, 1.13), handlelength=1.0, columnspacing=1.0)
    axa.text(-1.6 / 2.0 + .005, 1.035, 'filled, significant; open, not significant', transform=axa.transAxes, ha='left', va='bottom',
             fontsize=6, color=DGREY)
    # b: heterogeneity left after removal at the statement (mediate_summary_v6.csv, loc = stmt)
    M = pd.read_csv(A / "mediate_summary_v6.csv"); M["k"] = M.kind.str.replace(r"\d", "", regex=True)
    JB = [("V840_mediate_qwen36", "Qwen3.6 27B"), ("V842_mediate_olmo32", "OLMo 3.1 32B"), ("V843_mediate_gemma4_26b_a4b", "Gemma 4 26B")]
    CH = [("thaler_costs_justify", "costs justify prices"), ("freedman_consistency", "consistency")]
    axb = axin(fig, 5.02, top, 1.93, h); y = 0.0; ticks, labs, hdrs, rows = [], [], [], {}
    for jb, lab in JB:
        if ticks: y += .5
        ticks.append(y); labs.append(lab); hdrs.append(len(ticks) - 1); y += 1
        for t, tl in CH:
            ticks.append(y); labs.append(tl); rows[(jb, t)] = y; y += 1.15
    axb.axvline(100, color=GREY, lw=0.7, ls=DOT, zorder=0)
    vb = {}
    for (jb, t), yy in rows.items():
        g = M[(M.job == jb) & (M.test == t) & (M["loc"] == "stmt")]
        o = 100 * g[g.k == "own"].share_left.values; rn = 100 * g[g.k == "rand"].share_left.values; ot = 100 * g[g.k == "other"].share_left.values
        vb[f'{jb} {t}'] = {'own': o.round(1).tolist(), 'random': rn.round(1).tolist(), 'other': ot.round(1).tolist()}
        axb.scatter(o, [yy - .3] * len(o), s=13, color=NAVY, lw=0, zorder=3)
        axb.scatter(rn, [yy] * len(rn), s=9, color=GREY, lw=0, zorder=3)
        axb.scatter(ot, [yy + .3] * len(ot), s=9, facecolor='white', edgecolor=DGREY, lw=0.6, zorder=3)
    axb.set_yticks(ticks); axb.set_yticklabels(labs); axb.set_ylim(y - 1.15 + .7, -.7)
    for i in hdrs: axb.get_yticklabels()[i].set_fontstyle('italic'); axb.get_yticklabels()[i].set_color(DGREY)
    axb.tick_params(axis='y', length=0, pad=2); axb.spines['left'].set_visible(False)
    axb.set_xlim(-5, 130); axb.set_xticks([0, 25, 50, 75, 100, 125])
    axb.set_xlabel('Heterogeneity left after removal (%)')
    axb.legend([Line2D([], [], ls='none', marker='o', ms=3.6, mfc=NAVY, mec=NAVY), Line2D([], [], ls='none', marker='o', ms=3.0, mfc=GREY, mec=GREY),
                Line2D([], [], ls='none', marker='o', ms=3.0, mfc='white', mec=DGREY, mew=0.6)],
               ["characteristic's representation", 'random directions', 'other characteristics'], ncol=1, loc='lower left',
               bbox_to_anchor=(-.55, 1.04), handlelength=1.0, labelspacing=.2)
    panel(fig, 0, 0.02, 'a', 'Heterogeneity by described characteristic')
    panel(fig, 3.86, 0.02, 'b', "Removal of the characteristic's representation")
    VALS['fig7'] = {'a_interaction_dz (DEVLOG 607 via RESULTS_INVENTORY 5; n = 18, filled if |t| > %.3f)' % tcrit: VA,
                    'a_significant': {f'{m} | {t}': s_ for (m, t), s_ in sig.items()},
                    'b_share_left_pct (mediate_summary_v6.csv, loc=stmt)': vb}
    save(fig, 'fig7')


# ================================================================ Fig. 7  training of OLMo 3 7B
def xbreak(ax, x, size=0.012):
    """Two short slanted marks on the x axis at data position x (axis break)."""
    tr = ax.get_xaxis_transform()
    for dx in (-0.06, 0.06):
        ax.plot([x + dx - .05, x + dx + .05], [-size * 2, size * 2], transform=tr, color=INK, lw=0.6, clip_on=False, zorder=5)
    ax.add_patch(Rectangle((x - .055, -0.01), .11, .02, transform=tr, fc='white', ec='none', clip_on=False, zorder=4))


def fig7():
    has_d = CKPT_JSON.exists()
    H = 4.5
    fig = plt.figure(figsize=(W, H))
    TR = pd.read_csv(A / "olmo7_training_v1_read.csv"); T1 = TR[TR.stage == 1]
    CA = pd.read_csv(A / "olmo7_training_v4a.csv").set_index("ckpt")
    lg = lambda s: np.log10(s) if s > 0 else 2.0             # step 0 drawn at the position of 100 updates, after an axis break
    pa, pb, pc, pd_ = (0.5, 0.36, 2.95, 1.45), (4.2, 0.36, 2.78, 1.45), (0.5, 2.68, 2.95, 1.45), (4.2, 2.68, 2.78, 1.45)
    # a: identification and meaning over updates
    axa = axin(fig, *pa)
    st = T1.step.values; xa = np.array([lg(s) for s in st])
    ck = {"V7500_olmo7_stage1_step0": 0, "V7501_olmo7_stage1_step1000": 1000, "V7502_olmo7_stage1_step3000": 3000, "V7503_olmo7_stage1_step10000": 1e4,
          "V7504_olmo7_stage1_step30000": 3e4, "V7506_olmo7_stage1_step300000": 3e5, "V7509_olmo7_stage1_step1413814": 1413814}
    xm = np.array([lg(s) for s in ck.values()]); mm = np.array([float(RD[k]) for k in ck])
    XF, XI = 6.85, 7.45
    idn_final = float(TR[TR.checkpoint == "main"].acc_layer16.iloc[0]); idn_inst = IDN["OLMo 3 7B"]
    m_final, m_inst = float(RD["V7512_olmo7_main"]), float(RD["V736_olmo7"])
    axa.axhline(.666, color=GREY, lw=0.7, ls=(0, (3, 1.6)), zorder=1)
    axa.text(6.25, .69, 'single words', ha='right', va='bottom', fontsize=6, color=GREY)
    axa.axhline(.10, color=GREY, lw=0.7, ls=DOT, zorder=1); chance_text(axa, 6.25, .075, ha='right', va='top', s='chance (meaning)')
    axa.plot(xa, T1.acc_layer16.values, '-o', color=NAVY, ms=3, lw=0.9, mew=0, zorder=3)
    axa.plot(xm, mm, '-o', color=NAVY, ms=3, lw=0.9, mfc='white', mew=0.8, zorder=3)
    axa.plot([XF, XI], [idn_final, idn_inst], 'o', color=NAVY, ms=3, mew=0, zorder=3)
    axa.plot([XF, XI], [m_final, m_inst], 'o', color=NAVY, ms=3, mfc='white', mew=0.8, zorder=3)
    axa.text(4.55, .935, 'construct (of 66)', ha='center', va='bottom', fontsize=6, color=NAVY)
    axa.text(4.95, .515, 'meaning (of 10 definitions)', ha='center', va='top', fontsize=6, color=NAVY)
    axa.set_xlim(1.75, 7.7); axa.set_ylim(0, 1); axa.set_yticks([0, .2, .4, .6, .8, 1]); axa.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    axa.set_xticks([2, 3, 4, 5, 6, XF, XI]); axa.set_xticklabels(['0', '1k', '10k', '100k', '1M', 'final', 'Instruct'])
    xbreak(axa, 2.5); xbreak(axa, 6.5)
    axa.set_xlabel('Updates (first training stage)'); axa.set_ylabel('Proportion identified')
    # b, c, d: categorical positions after the break
    XS = {"10k": 4.0, "300k": np.log10(3e5), "1.41M": np.log10(1413814), "main": 6.95, "SFT": 7.45, "DPO": 7.95}
    TK = (list(XS.values()), ['10k', '300k', '1.41M', 'final', 'SFT', 'DPO'])

    def cat_axis(ax):
        ax.set_xlim(3.7, 8.2); ax.set_xticks(TK[0]); ax.set_xticklabels(TK[1]); xbreak(ax, 6.55)
        ax.set_xlabel('Updates (first training stage), then tuning')

    def series(ax, col, color, mfc=None, marker='o'):
        y1 = [CA.loc[k, col] for k in ("10k", "300k", "1.41M")]; y2 = [CA.loc[k, col] for k in ("main", "SFT", "DPO")]
        kw = dict(color=color, ms=3, lw=0.9, marker=marker, mew=0 if mfc is None else 0.8, mfc=mfc or color, zorder=3)
        ax.plot([XS[k] for k in ("10k", "300k", "1.41M")], y1, '-', **kw); ax.plot([XS[k] for k in ("main", "SFT", "DPO")], y2, '-', **kw)
        return [float(v) for v in y1 + y2]
    axb = axin(fig, *pb); axb.axhline(0, color=GREY, lw=0.5, zorder=0)
    rb = series(axb, "r_nat_push", NAVY); cat_axis(axb)
    axb.set_ylim(-.1, .7); axb.set_yticks([0, .2, .4, .6]); axb.set_yticklabels(['0', '0.2', '0.4', '0.6'])
    axb.set_ylabel("Induced vs narratives' effects\non judgments (r)")
    axc = axin(fig, *pc)
    # c: difference in the judgment of the construct the situations differ in (own item, diagonal of the natural matrix), as the text reports
    OWNJ = {"10k": "V820_src4a_olmo7_stage1_step10000", "300k": "V821_src4a_olmo7_stage1_step300000", "1.41M": "V822_src4a_olmo7_stage1_step1413814",
            "main": "V823_src4a_olmo7_main", "SFT": "V824_src4a_olmo7_sft", "DPO": "V825_src4a_olmo7_dpo"}
    CA["natural_own"] = pd.Series({k: float(np.mean(list(json.load(open(RV / j / "V4a_causal" / "summary.json"))["natural_own"].values()))) for k, j in OWNJ.items()})
    rc = series(axc, "natural_own", NAVY); cat_axis(axc)
    axc.set_ylim(0, 1.3); axc.set_yticks([0, .4, .8, 1.2]); axc.set_yticklabels(['0', '0.4', '0.8', '1.2'])
    axc.set_ylabel('Judgment difference between\nnarratives (scale points)')
    VALS['figS1'] = {'a_identification_layer16 (olmo7_training_v1_read.csv stage 1)': dict(zip([int(s) for s in st], T1.acc_layer16.round(4).tolist())),
                    'a_identification_final_main_layer16': idn_final, 'a_identification_Instruct (IDN, best middle layer)': idn_inst,
                    'a_meaning (readout_judge_v6_all_models.csv)': dict(zip([int(s) for s in ck.values()], mm.round(3).tolist())),
                    'a_meaning_final_V7512': m_final, 'a_meaning_Instruct_V736': m_inst, 'a_single_words': .666, 'a_chance_meaning': .10,
                    'b_r_nat_push (olmo7_training_v4a.csv)': dict(zip(TK[1], [float(v) for v in rb])),
                    'c_natural_own (V82x V4a_causal/summary.json natural_own, mean over 66)': dict(zip(TK[1], [float(v) for v in rc]))}
    panel(fig, 0, 0.02, 'a', 'Identification over training')
    panel(fig, pb[0] - 0.5, 0.02, 'b', 'Induced representations and judgments')
    panel(fig, 0, pc[1] - 0.4, 'c', 'Judgment differences between narratives')
    if has_d:                                                   # d: Study 3 over the checkpoints
        V = json.load(open(CKPT_JSON))
        XD = {"step 1k": (3.0, '1k'), "step 3k": (np.log10(3e3), '3k'), "step 10k": (4.0, '10k'), "step 30k": (np.log10(3e4), '30k'),
              "step 300k": (np.log10(3e5), '300k'), "step 1.41M": (np.log10(1413814), '1.41M'),
              "base (end of training)": (6.95, 'final'), "SFT": (7.5, 'SFT'), "DPO": (8.05, 'DPO'), "Instruct": (8.65, 'Instruct')}
        missing = [k for k in XD if k not in V]
        if missing: print('  !! Fig. 7d: stages missing in', CKPT_JSON.name, missing)
        axd = axin(fig, *pd_)
        axd.set_xlim(2.75, 8.95); axd.set_xticks([XD[k][0] for k in XD]); axd.set_xticklabels([XD[k][1] for k in XD]); xbreak(axd, 6.55)
        axd.set_xlabel('Updates (first training stage), then tuning')
        pts = [(XD[k][0], V[k]) for k in XD if k in V]

        def get(v, f):
            s = v.get(f); return None if s is None else int(str(s).split('/')[0])
        for nm, f, c, kw in (('representation', 'own_shift_pos', NAVY, dict(mfc=NAVY, mew=0)), ('answer', 'response_sig', RESP, dict(mfc='white', mew=0.8))):
            xy = [(x, get(v, f)) for x, v in pts if get(v, f) is not None]
            seg1 = [p for p in xy if p[0] < 6.55]; seg2 = [p for p in xy if p[0] > 6.55]
            for sg in (seg1, seg2):
                if sg: axd.plot(*zip(*sg), '-o', color=c, ms=3, lw=0.9, mec=c, zorder=3, **kw)
        axd.axhline(30, color=GREY, lw=0.7, ls=DOT, zorder=1); chance_text(axd, 2.8, 30.8, ha='left', va='bottom')
        axd.text(4.65, 48.5, "activation of the mediator's representation", ha='center', va='bottom', fontsize=6, color=NAVY)
        axd.text(5.55, 22.5, 'original effect in the responses', ha='center', va='top', fontsize=6, color=RESP)
        axd.set_ylim(0, 60); axd.set_yticks([0, 15, 30, 45, 60]); axd.set_ylabel('Experiments (of 60)')
        panel(fig, pd_[0] - 0.5, pd_[1] - 0.4, 'd', 'Experiments over training')
        VALS['figS1']['d_v3_ckpt (%s): representation own_shift_pos / answers response_sig' % CKPT_JSON.name] = {
            XD[k][1]: [get(V[k], 'own_shift_pos'), get(V[k], 'response_sig')] for k in XD if k in V}
    save(fig, 'figS1')


if __name__ == "__main__":
    import sys                                                  # python make_figs_ps.py [figure numbers 1-8; 8 = Fig. S1]; default all
    import fig_measure_ps
    FUN = (lambda: fig_measure_ps.fig_measure(sys.modules[__name__]), fig2, fig3, fig4, fig1, fig5, fig6, fig7)   # fig1() draws Fig. 5
    sel = [int(a) for a in sys.argv[1:]] or list(range(1, 9))
    for i in sel: FUN[i - 1]()
    pv = OUT / "plotted_values.json"
    allv = json.load(open(pv, encoding="utf-8")) if (pv.exists() and len(sel) < 8) else {}
    allv.update(json.loads(json.dumps(VALS, default=str)))      # a partial run replaces only the figures it drew
    json.dump(allv, open(pv, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("figures", sel, "and plotted_values.json written to", OUT)
