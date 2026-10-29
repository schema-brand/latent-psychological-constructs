"""Fig. 1 of the Psychological Science version (PI 22 Oct: the old Fig. 1a, made into its own figure that explains visually how we
measure a construct's representation, its activation, and the related measures). Imported by make_figs_ps.py, which supplies the
style constants and helpers. Panels: a pairs of narratives from six writers; b from a narrative to its internal state; c observation
and representation; d construct validity (identification of a held-out writer's pairs; multitrait-multimethod correlations);
e activation of a representation; f the manipulation's activation of the mediator's representation. Only the numbers in d are
results (identification at layers 30-50%, word sequences, chance, MTMM ranges; the same sources as the old Fig. 1a); every other
cell, dot and bar is a schematic illustration and is recorded as such in plotted_values.json."""
import json
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def fig_measure(M):
    W, A, NAVY, LBLUE, GREY, LGREY, DGREY, INK, MGREY, RESP = (M.W, M.A, M.NAVY, M.LBLUE, M.GREY, M.LGREY, M.DGREY, M.INK, M.MGREY,
                                                              M.RESP)
    H = 7.3
    DIV = LinearSegmentedColormap.from_list('greynavy', ['#9a9a9a', '#ffffff', NAVY])   # signed values; navy = representations
    MT_OLD = {k: plt.rcParams[k] for k in ('mathtext.default', 'mathtext.fontset', 'mathtext.rm', 'mathtext.it', 'mathtext.bf')}
    plt.rcParams.update({'mathtext.default': 'it', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
                         'mathtext.bf': 'Arial:bold'})                       # R47: formal notation in the panels, as in the Method
    fig = plt.figure(figsize=(W, H), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    Y = lambda t: H - t
    rng = np.random.default_rng(7)

    def T(x, t, s, **kw):
        kw.setdefault('fontsize', 6.5); kw.setdefault('zorder', 20); return ax.text(x, Y(t), s, **kw)

    def rbox(x, t, w, h, fc, ec='none', lw=0.6, r=0.035, z=2):
        ax.add_patch(FancyBboxPatch((x, Y(t + h)), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))

    def arr(x0, t0, x1, t1, color=INK, lw=0.7, ms=6, z=12, cs=None):
        kw = dict(arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z)
        if cs: kw['connectionstyle'] = cs
        ax.add_patch(FancyArrowPatch((x0, Y(t0)), (x1, Y(t1)), **kw))

    def column(x, tc, vals, cmap, w=0.12, ch=0.07, z=12, vmin=0.0, vmax=1.0):
        cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap; t0 = tc - len(vals) * ch / 2
        for k, v in enumerate(vals):
            c = cm((v - vmin) / (vmax - vmin))
            ax.add_patch(Rectangle((x, Y(t0 + (k + 1) * ch)), w, ch, fc=c, ec='white', lw=0.4, zorder=z))

    def head(x, t, s, title):
        T(x, t, s, fontsize=8, fontweight='bold', va='top')
        T(x + 0.155, t + 0.012, title, fontsize=7, va='top')

    XR = 3.62                                                     # left edge of the right-hand panels
    for t_ in (2.83, 4.98):                                       # row dividers
        ax.plot([0, W], [Y(t_ - 0.09)] * 2, color=LGREY, lw=0.6, zorder=1)
    ax.plot([XR - 0.1, XR - 0.1], [Y(0.03), Y(H - 0.03)], color=LGREY, lw=0.6, zorder=1)

    # ===================== a: pairs of narratives =====================================================================
    head(0, 0.03, 'a', 'Pairs of narratives from six writers')
    x0, cw, dx, dy, nb = 0.03, 2.45, 0.065, 0.1, 5
    tf = 0.36 + nb * dy; chh = 1.42
    for k in range(nb, 0, -1):
        rbox(x0 + k * dx, tf - k * dy, cw, chh, '#f6f6f6', MGREY, lw=0.5, z=2 + nb - k)
        T(x0 + k * dx + 0.06, tf - k * dy + dy / 2, f'writer {k + 1}', va='center', fontsize=6, color=DGREY)
    rbox(x0, tf, cw, chh, 'white', GREY, lw=0.6, z=9)
    T(x0 + 0.06, tf + dy / 2 + 0.005, 'writer 1', va='center', fontsize=6, color=DGREY)
    hi = ("The bakery that delivers rolls to the office\nkitchen each morning has started to deliver\n"
          "fruit loaves too. They are baked in the same\novens by the same bakers from the same flour …")
    lo = ("The bakery that delivers rolls to the office\nkitchen each morning has started to deliver\n"
          "bicycle lights too. They are assembled by newly\nhired workers with soldering irons …")
    for k, (hd, body, fc, hc) in enumerate((('High narrative', hi, LBLUE, NAVY), ('Low narrative', lo, LGREY, DGREY))):
        tb_ = tf + 0.13 + k * 0.63
        rbox(x0 + 0.05, tb_, cw - 0.1, 0.59, fc, z=10)
        T(x0 + 0.1, tb_ + 0.045, hd, va='top', color=hc, fontweight='bold', fontsize=6)
        T(x0 + 0.1, tb_ + 0.155, body, va='top', fontsize=6, linespacing=1.12)
    T(x0 + cw / 2, tf - nb * dy - 0.01, 'Example: perceived fit of a new product', ha='center', va='bottom', fontsize=6, color=DGREY)
    T(0.03, tf + chh + 0.1, 'For each of 66 constructs: pairs that describe the same scenario and differ only in the\n'
      'construct; two rater LLMs checked that the high narrative is rated higher (8,311 pairs)',
      va='top', fontsize=6, color=DGREY, linespacing=1.15)

    # ===================== b: from a narrative to its internal state ==================================================
    head(XR, 0.03, 'b', "From a narrative to its internal state")
    toks = ['The', 'bakery', '…', 'bicycle', 'lights']
    tx0, tw, tg = 4.3, 0.36, 0.05
    xt = [tx0 + k * (tw + tg) for k in range(len(toks))]; xe = xt[-1] + tw
    tt = 2.32                                                     # top of the word-fragment boxes
    for x, s in zip(xt, toks):
        rbox(x, tt, tw, 0.2, 'white', MGREY, lw=0.5, z=10); T(x + tw / 2, tt + 0.1, s, ha='center', va='center', fontsize=6)
    T((tx0 + xe) / 2, tt + 0.28, 'word fragments of one narrative', ha='center', va='top', fontsize=6, color=DGREY)
    nl = 6; lt0, lh, lg = 0.66, 0.16, 0.09                        # layer bands, top band = last layer (room above for the formula)
    tops = [lt0 + k * (lh + lg) for k in range(nl)]
    for x in xt:                                                  # each word fragment passes through every layer
        ax.plot([x + tw / 2] * 2, [Y(tt), Y(tops[0])], color=MGREY, lw=0.5, zorder=3)
    for k, t_ in enumerate(tops):
        sel = k in (3, 4)
        rbox(tx0 - 0.04, t_, xe - tx0 + 0.08, lh, M.LBLUE if sel else LGREY, NAVY if sel else 'none', lw=0.6, z=4)
        for x in xt:
            ax.add_patch(Rectangle((x + tw / 2 - 0.05, Y(t_ + lh / 2 + 0.035)), 0.1, 0.07, fc=NAVY if sel else MGREY, ec='white',
                                   lw=0.3, zorder=6))
    T(tx0 - 0.1, tops[0] + lh / 2, 'last layer', ha='right', va='center', fontsize=6, color=DGREY)
    T(tx0 - 0.1, tops[-1] + lh / 2, 'layer 1', ha='right', va='center', fontsize=6, color=DGREY)
    T(tx0 - 0.1, (tops[3] + tops[4] + lh) / 2, 'layers $l \\in B$\n(30–50%)', ha='right', va='center', fontsize=6, color=NAVY, linespacing=1.1)
    T((tx0 + xe) / 2, tops[0] - 0.07, r'residual stream $\mathbf{h}_{lt}(x)$ of word fragment $t$ at layer $l$', ha='center', va='bottom',
      fontsize=6, color=DGREY)
    T((tx0 + xe) / 2, 0.33, r'$\mathbf{h}(x)=\frac{1}{|B|\,T}\sum_{l\in B}\;\sum_{t=1}^{T}\mathbf{h}_{lt}(x)$', ha='center', va='center',
      fontsize=7.5, color=NAVY)
    tcm = (tops[3] + tops[4] + lh) / 2
    arr(xe + 0.06, tcm, 6.43, tcm, color=NAVY)
    vb = rng.uniform(.2, .9, 7)
    column(6.47, tcm, vb, 'Blues', w=0.13)
    T(6.535, tcm + 0.32, "the narrative's\ninternal state\n" + r'$\mathbf{h}(x)$', ha='center', va='top', fontsize=6,
      linespacing=1.12, color=NAVY)

    # ===================== c: observation and representation ==========================================================
    tc_ = 2.83
    head(0, tc_, 'c', 'Observation and representation of a construct')
    ce = tc_ + 0.64                                               # centre of the columns
    hv, lv = rng.uniform(.25, .9, 7), rng.uniform(.1, .7, 7)
    column(0.12, ce, hv, 'Blues'); T(0.18, ce + 0.3, 'high\n' + r'$\mathbf{z}(x_i^{\mathrm{high}})$', ha='center', va='top', fontsize=6, linespacing=1.1)
    T(0.36, ce, '−', ha='center', va='center', fontsize=8)
    column(0.47, ce, lv, 'Blues'); T(0.53, ce + 0.3, 'low\n' + r'$\mathbf{z}(x_i^{\mathrm{low}})$', ha='center', va='top', fontsize=6, linespacing=1.1)
    T(0.71, ce, '=', ha='center', va='center', fontsize=8)
    column(0.82, ce, hv - lv, DIV, vmin=-0.9, vmax=0.9); T(0.88, ce + 0.3, 'observation\n' + r'$\mathbf{o}_i$', ha='center', va='top',
                                                           fontsize=6, linespacing=1.1)
    arr(1.03, ce, 1.2, ce)
    for k in range(4):                                            # observations of one construct from five writers' pairs
        column(1.25 + k * 0.05, ce - 0.06 + k * 0.03, (hv - lv) * rng.uniform(.6, 1.2, 7) + rng.normal(0, .1, 7), DIV,
               vmin=-0.9, vmax=0.9, z=12 + k)
    T(1.37, ce + 0.33, 'pairs of\nconstruct $c$', ha='center', va='top', fontsize=6, linespacing=1.1)
    arr(1.55, ce, 1.8, ce)
    column(1.85, ce, np.clip((hv - lv) * 1.3, -0.85, 0.85), DIV, vmin=-0.9, vmax=0.9, w=0.13)
    T(1.915, ce + 0.33, 'represen-\ntation\n' + r'$\mathbf{r}_c$', ha='center', va='top', fontsize=6, linespacing=1.1, color=NAVY)
    T(0.03, tc_ + 1.42, r'$\mathbf{z}(x)=\mathbf{h}(x)\oslash\mathbf{s}$' + ' (standardized)', va='center', fontsize=7, color=INK)
    T(0.03, tc_ + 1.63, r'$\mathbf{o}_i=\mathbf{z}(x_i^{\mathrm{high}})-\mathbf{z}(x_i^{\mathrm{low}})$', va='center', fontsize=7.5, color=INK)
    T(0.03, tc_ + 1.86, r'$\mathbf{r}_c=\frac{1}{n_c}\sum_{i\in I_c}\mathbf{o}_i$', va='center', fontsize=7.5, color=NAVY)
    # geometric view: two dimensions of the internal states
    axg = M.axin(fig, 2.35, tc_ + 0.3, 1.05, 1.05)
    lo_pts = rng.normal([0.32, 0.35], 0.07, (14, 2)); hi_pts = rng.normal([0.68, 0.66], 0.07, (14, 2))
    axg.scatter(*lo_pts.T, s=5, color=GREY, lw=0, zorder=3); axg.scatter(*hi_pts.T, s=5, color=NAVY, lw=0, zorder=3)
    ml, mh = lo_pts.mean(0), hi_pts.mean(0)
    axg.annotate('', xy=mh, xytext=ml, arrowprops=dict(arrowstyle='-|>', color=NAVY, lw=1.0, mutation_scale=7), zorder=5)
    axg.set_xlim(0, 1); axg.set_ylim(0, 1); axg.set_xticks([]); axg.set_yticks([])
    axg.text(0.03, 0.97, 'high', color=NAVY, fontsize=6, va='top'); axg.text(0.03, 0.86, 'low', color=GREY, fontsize=6, va='top')
    T(2.85, tc_ + 1.42, 'two dimensions of the\ninternal states; arrow: ' + r'$\mathbf{r}_c$', ha='center', va='top', fontsize=6,
      linespacing=1.1, color=DGREY)

    # ===================== d: construct validity ======================================================================
    head(XR, tc_, 'd', 'Validity of construct representations')
    column(3.75, ce, hv - lv + rng.normal(0, .12, 7), DIV, vmin=-0.9, vmax=0.9)
    T(3.81, ce + 0.3, "held-out\nwriter's\npair " + r'$\mathbf{o}_i$', ha='center', va='top', fontsize=6, linespacing=1.1)
    arr(3.92, ce, 4.1, ce)
    axs = M.axin(fig, 4.18, tc_ + 0.35, 1.15, 0.62)
    sims = rng.uniform(.03, .2, 16); sims[9] = .48
    axs.bar(range(16), sims, width=0.7, color=[NAVY if k == 9 else M.MGREY for k in range(16)], lw=0)
    axs.set_xticks([]); axs.set_yticks([]); axs.set_xlim(-0.8, 15.8); axs.set_ylim(0, 0.55)
    axs.text(9, 0.5, 'own\nconstruct', ha='center', va='bottom', fontsize=6, color=NAVY, linespacing=1.0)
    T(4.75, tc_ + 1.02, r'$\cos(\mathbf{o}_i,\mathbf{r}_c^{(-w)})$, $c=1,\ldots,66$', ha='center', va='top', fontsize=6, color=DGREY)
    T(4.75, tc_ + 1.24, r'$\hat{c}_i=\mathrm{arg\,max}_c\;\cos(\mathbf{o}_i,\mathbf{r}_c^{(-w)})$', ha='center', va='top', fontsize=7, color=NAVY)
    b1 = M.s3_band_b1(); idn_lo, idn_hi = min(b1.values()) / 100, max(b1.values()) / 100
    NG = json.load(open(A / "ngram_floor_v6.json")); w3 = NG["strongest_floor"]["first"]
    T(3.65, tc_ + 1.52, f"Identification of the held-out writer's\npairs: {idn_lo * 100:.0f}–{idn_hi * 100:.0f}% (16 LLMs); word\n"
      f"sequences {w3 * 100:.0f}%; chance 1.5%", va='top', fontsize=6, color=NAVY, linespacing=1.15)
    cs_, mx0, mtop = 0.1, 6.12, tc_ + 0.4                        # multitrait-multimethod sketch (constructs A-C x writers 1-2)
    for i in range(6):
        for j in range(i + 1):
            if i == j: fc, ec = 'white', MGREY
            elif i % 3 == j % 3: fc, ec = NAVY, 'white'
            else: fc, ec = '#d9d9d9', 'white'
            ax.add_patch(Rectangle((mx0 + j * cs_, Y(mtop + (i + 1) * cs_)), cs_, cs_, fc=fc, ec=ec, lw=0.6, zorder=12))
    mbot = mtop + 6 * cs_
    for k in range(6):
        T(mx0 + (k + .5) * cs_, mbot + 0.02, 'ABC'[k % 3], ha='center', va='top', fontsize=6)
        T(mx0 - 0.03, mtop + (k + .5) * cs_, 'ABC'[k % 3], ha='right', va='center', fontsize=6)
    for w_ in range(2):
        xa, xb = mx0 + w_ * 3 * cs_ + 0.012, mx0 + (w_ + 1) * 3 * cs_ - 0.012
        ax.plot([xa, xa, xb, xb], [Y(mbot + .125), Y(mbot + .15), Y(mbot + .15), Y(mbot + .125)], color=DGREY, lw=0.5, zorder=12)
        T((xa + xb) / 2, mbot + .17, f'writer {w_ + 1}', ha='center', va='top', fontsize=6, color=DGREY)
    T(mx0 + 3 * cs_, mtop - 0.06, 'multitrait-multimethod\n' + r'$\rho(\mathbf{m}_{cw},\mathbf{m}_{c^{\prime}w^{\prime}})$', ha='center', va='bottom',
      fontsize=6, color=DGREY, linespacing=1.2)
    conv = [v[0] for v in M.MTMM.values()]; disc = [x for v in M.MTMM.values() for x in v[1:]]
    T(mx0 + 3 * cs_, mbot + 0.38, f'same construct, different\nwriters: r = {M.r2s(min(conv))}–{M.r2s(max(conv))}', ha='center', va='top',
      fontsize=6, color=NAVY, linespacing=1.12)
    T(mx0 + 3 * cs_, mbot + 0.66, f'different constructs:\nr = {M.r2s(min(disc))}–{M.r2s(max(disc))}', ha='center', va='top',
      fontsize=6, color=GREY, linespacing=1.12)

    # ===================== e: activation of a representation ==========================================================
    te = 4.98
    head(0, te, 'e', 'Activation of a representation')
    axp = M.axin(fig, 0.25, te + 0.3, 1.75, 1.7)
    ang = np.deg2rad(22); u = np.array([np.cos(ang), np.sin(ang)]); o = np.array([0.08, 0.1])
    h_ = np.array([0.55, 0.82]); pj = o + u * ((h_ - o) @ u)
    axp.plot(*zip(o - u * 0.02, o + u * 1.0), color=NAVY, lw=0.6, ls=(0, (3, 2)), zorder=2)
    axp.annotate('', xy=o + u * 0.32, xytext=o, arrowprops=dict(arrowstyle='-|>', color=NAVY, lw=1.1, mutation_scale=7), zorder=4)
    axp.annotate('', xy=h_, xytext=o, arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.9, mutation_scale=7), zorder=4)
    axp.plot(*zip(h_, pj), color=GREY, lw=0.6, ls=(0, (2, 1.5)), zorder=3)
    axp.plot(*zip(o, pj), color=NAVY, lw=3.0, alpha=0.35, solid_capstyle='butt', zorder=3)
    axp.set_xlim(0, 1.05); axp.set_ylim(0, 1.0); axp.axis('off')
    axp.text(h_[0] + 0.03, h_[1] + 0.02, "internal state of a text\n" + r'$\mathbf{z}(x)$', fontsize=6, va='bottom', ha='left', linespacing=1.1)
    axp.text(*(o + u * 0.33 + np.array([0.0, -0.1])), 'representation, unit length\n' + r'$\mathbf{r}_c/\Vert\mathbf{r}_c\Vert$', fontsize=6, color=NAVY,
             va='top', ha='left', linespacing=1.1)
    axp.text(*(pj + np.array([0.04, -0.03])), 'activation ' + r'$a_c(x)$', fontsize=6, color=NAVY, va='top', ha='left', fontweight='bold')
    T(2.05, te + 0.55, r'$a_c(x)=\mathbf{z}(x)\cdot\dfrac{\mathbf{r}_c}{\Vert\mathbf{r}_c\Vert}$', va='center', fontsize=8.5, color=NAVY)
    T(2.05, te + 0.85, "for any text $x$ that the LLM\nprocesses, such as an\nexperiment's scenario\ndescription", va='top', fontsize=6,
      linespacing=1.15, color=DGREY)

    # ===================== f: the manipulation's activation of the mediator's representation ===========================
    head(XR, te, 'f', "Manipulation's activation of the mediator's representation")
    rbox(3.72, te + 0.38, 1.45, 0.3, M.LBLUE, NAVY, lw=0.5, z=10)
    T(4.445, te + 0.53, r'treatment $x^{\mathrm{T}}$: Sam is a grown man', ha='center', va='center', fontsize=6)
    rbox(3.72, te + 0.76, 1.45, 0.3, 'white', MGREY, lw=0.5, z=10)
    T(4.445, te + 0.91, r'control $x^{\mathrm{C}}$: Sam is a baby', ha='center', va='center', fontsize=6)
    T(4.445, te + 1.16, '(moral typecasting; mediator:\nmoral agency)', ha='center', va='top', fontsize=6, color=DGREY, linespacing=1.1)
    arr(5.2, te + 0.72, 5.42, te + 0.72)
    axd = M.axin(fig, 5.62, te + 0.38, 1.35, 0.98)
    oth = rng.normal(0.12, 0.09, 14); medv = 0.95
    axd.bar([0], [medv], width=0.7, color=NAVY, lw=0)
    axd.bar(np.linspace(1.2, 3.8, 14), oth, width=0.13, color=M.MGREY, lw=0)
    axd.axhline(oth.mean(), xmin=0.27, xmax=0.98, color=DGREY, lw=0.6, ls=(0, (2, 1.5)))
    axd.axhline(0, color=INK, lw=0.5)
    axd.annotate('', xy=(0.55, medv), xytext=(0.55, oth.mean()), arrowprops=dict(arrowstyle='<|-|>', color=INK, lw=0.6, mutation_scale=5))
    axd.set_xlim(-0.5, 4.0); axd.set_ylim(-0.2, 1.1); axd.set_xticks([0, 2.5]); axd.set_xticklabels(['mediator $m$', r'65 other constructs $c\neq m$'])
    axd.text(0.68, (medv + oth.mean()) / 2, r'$\Delta_m$', fontsize=7, va='center', ha='left', color=INK)
    axd.set_yticks([]); axd.spines['left'].set_visible(False); axd.spines['bottom'].set_visible(False)
    axd.tick_params(axis='x', length=0, pad=2)
    T(5.6, te + 0.29, r'$a_c(x^{\mathrm{T}})-a_c(x^{\mathrm{C}})$', ha='left', va='bottom', fontsize=6.5, color=DGREY)
    T(3.65, te + 1.78, r'$\Delta_m=\sigma_m\left\{[a_m(x^{\mathrm{T}})-a_m(x^{\mathrm{C}})]-\dfrac{1}{65}\sum_{c\neq m}[a_c(x^{\mathrm{T}})-a_c(x^{\mathrm{C}})]\right\}$',
      va='center', fontsize=7.5, color=NAVY)
    T(3.65, te + 2.08, r'$\sigma_m=\pm 1$: direction predicted by the theory; reported in ' + M.DZ + '\nacross seven sets of scenario descriptions',
      va='center', fontsize=6, color=DGREY, linespacing=1.15)
    T(0.03, H - 0.05, 'Schematic, except the results in d', ha='left', va='bottom', fontsize=6, color=GREY)

    M.VALS['fig1'] = {'a': 'pairs_v6.json claude__perceived_fit__L1_0__1 (shortened); 8,311 pairs (Method)',
                      'd identification range over the 16 LLMs at layers 30-50% (Table S3 Panel A)': [idn_lo, idn_hi],
                      'd words (ngram_floor_v6.json strongest_floor)': w3, 'd chance': 1 / 66,
                      'd MTMM same construct, different writers (min, max over 16 LLMs)': [min(conv), max(conv)],
                      'd MTMM different constructs (min, max over 16 LLMs)': [min(disc), max(disc)],
                      'schematic': 'all cells, dots, bars and arrows in a-f other than the numbers in d are illustrations'}
    M.save(fig, 'fig1')
    plt.rcParams.update(MT_OLD)
