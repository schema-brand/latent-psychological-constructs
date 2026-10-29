"""Figure 1 of the latent paper since v49 (PI 29 Oct: "cool, abstract and funny": on the left a horse and amazed psychologists, on the
right a robot (an LLM) and psychologists who are just as amazed; PI: no annotations, more polished). A flat vector cartoon drawn here
(smooth outlines from Catmull-Rom splines), no photographs or generated images. Style constants and QA from make_figs_ps.py.
Usage: python fig_cleverhans.py -> figs_ps/fig_cleverhans.png/.pdf"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, FancyBboxPatch, PathPatch, Arc
from matplotlib.path import Path as MPath
import make_figs_ps as M

W, H = M.W, 3.0
INK, GREY, DGREY = M.INK, M.GREY, M.DGREY
SKIN, SKIN2, HAIR, HAIR2 = '#f2cba8', '#e2b48e', '#3b2f2a', '#8a5a3c'
HORSE, HORSE_FAR, MANE, HOOF = '#a0704a', '#86593a', '#4e3220', '#2f241b'
METAL, METAL_D, METAL_L, SCREEN, GLOW = '#d7dde4', '#a9b3be', '#eef2f6', '#1f2d3b', '#7fd6ff'
BOARD, WOOD = '#2c3a31', '#6b4a2f'
WOW = '#d9822b'
BG_A, BG_B = '#f7f1e6', '#eef3f8'

fig = plt.figure(figsize=(W, H), dpi=600)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off'); ax.set_aspect('equal')

def T(x, y, s, **kw):
    kw.setdefault('fontsize', 6.5); kw.setdefault('zorder', 40); kw.setdefault('va', 'center'); kw.setdefault('ha', 'center')
    return ax.text(x, y, s, **kw)
def poly(pts, fc, ec='none', lw=0.0, z=5): ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z, joinstyle='round'))
def circ(x, y, r, fc, ec='none', lw=0.0, z=6): ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))
def ell(x, y, w, h, fc, ec='none', lw=0.0, z=5, a=0, alpha=1): ax.add_patch(Ellipse((x, y), w, h, angle=a, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))
def seg(xs, ys, c=INK, lw=1.0, z=6): ax.plot(xs, ys, color=c, lw=lw, zorder=z, solid_capstyle='round', solid_joinstyle='round')
def rbox(x, y, w, h, fc, ec='none', lw=0.6, r=0.04, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))
def spline(pts, closed=True, n=24):
    """Catmull-Rom spline through the points: smooth outlines from a few control points."""
    P = np.asarray(pts, float); m = len(P); out = []
    rng = range(m) if closed else range(m - 1)
    for i in rng:
        p0, p1, p2, p3 = (P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]) if closed else \
            (P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, m - 1)])
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed: out.append(P[-1])
    return np.array(out)
def blob(pts, fc, ec='none', lw=0.0, z=5): poly(spline(pts), fc, ec=ec, lw=lw, z=z)
def stroke(pts, c, lw, z=5): s = spline(pts, closed=False); seg(s[:, 0], s[:, 1], c=c, lw=lw, z=z)
def taper(pts, w0, w1, fc, z=5):
    """A limb or tail: a centre line with a width that tapers from w0 to w1."""
    s = spline(pts, closed=False, n=16); d = np.gradient(s, axis=0); nrm = np.c_[-d[:, 1], d[:, 0]]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9; w = np.linspace(w0, w1, len(s))[:, None] / 2
    poly(np.r_[s + nrm * w, (s - nrm * w)[::-1]], fc, z=z)

def shadow(x, g, w): ell(x, g - 0.005, w, 0.05, '#000000', z=1, alpha=0.08)

def person(x, g, coat, era, s=1.0, glasses=False, mustache=False, pose='up', hair='short', hat=True, collar='#ffffff'):
    hip, sh, hy = g + 0.37 * s, g + 0.78 * s, g + 0.99 * s
    shadow(x, g, 0.36 * s)
    for sx in (-1, 1):                                                                     # legs and shoes
        seg([x + sx * 0.045 * s, x + sx * 0.055 * s], [hip, g + 0.03], c='#2a2a2a', lw=2.0, z=5)
        ell(x + sx * 0.075 * s, g + 0.02, 0.09 * s, 0.035, '#1c1c1c', z=6)
    blob([(x - 0.125 * s, sh), (x, sh + 0.02 * s), (x + 0.125 * s, sh), (x + 0.12 * s, hip + 0.02), (x, hip - 0.01), (x - 0.12 * s, hip + 0.02)], coat, z=6)
    poly([(x - 0.045 * s, sh + 0.008), (x + 0.045 * s, sh + 0.008), (x, sh - 0.13 * s)], collar, z=7)   # shirt V
    if era == 1904: poly([(x - 0.012, sh - 0.02), (x + 0.012, sh - 0.02), (x, sh - 0.09 * s)], '#7a1f1f', z=8)   # tie
    if pose == 'up':
        for sx in (-1, 1):
            taper([(x + sx * 0.11 * s, sh - 0.03), (x + sx * 0.22 * s, sh + 0.12 * s), (x + sx * 0.26 * s, sh + 0.29 * s)], 0.055 * s, 0.045 * s, coat, z=5)
            circ(x + sx * 0.262 * s, sh + 0.315 * s, 0.033 * s, SKIN, z=6)
    elif pose == 'cheeks':
        for sx in (-1, 1):
            taper([(x + sx * 0.11 * s, sh - 0.03), (x + sx * 0.17 * s, sh + 0.05), (x + sx * 0.105 * s, hy - 0.03)], 0.055 * s, 0.045 * s, coat, z=9)
            circ(x + sx * 0.1 * s, hy - 0.02, 0.033 * s, SKIN, z=10)
    elif pose == 'point':                                                                  # the questioner, pointing at the board
        taper([(x - 0.11 * s, sh - 0.03), (x - 0.15 * s, sh - 0.2 * s), (x - 0.13 * s, hip + 0.02)], 0.055 * s, 0.045 * s, coat, z=5)
        circ(x - 0.13 * s, hip, 0.03 * s, SKIN, z=6)
        taper([(x + 0.11 * s, sh - 0.03), (x + 0.27 * s, sh + 0.06 * s), (x + 0.42 * s, sh + 0.13 * s)], 0.055 * s, 0.045 * s, coat, z=5)
        circ(x + 0.43 * s, sh + 0.135 * s, 0.03 * s, SKIN, z=6)
    circ(x, hy, 0.125 * s, SKIN, z=8)                                                      # head, ears
    circ(x - 0.123 * s, hy, 0.025 * s, SKIN2, z=7); circ(x + 0.123 * s, hy, 0.025 * s, SKIN2, z=7)
    if hair == 'short' and not hat:
        ax.add_patch(Arc((x, hy + 0.012), 0.255 * s, 0.25 * s, theta1=8, theta2=172, color=HAIR, lw=3.0, zorder=9))
    elif hair == 'bun' and not hat:
        ax.add_patch(Arc((x, hy + 0.012), 0.255 * s, 0.25 * s, theta1=8, theta2=172, color=HAIR2, lw=3.0, zorder=9)); circ(x, hy + 0.15 * s, 0.045 * s, HAIR2, z=8)
    elif hair == 'grey' and not hat:
        ax.add_patch(Arc((x, hy + 0.012), 0.255 * s, 0.25 * s, theta1=8, theta2=172, color='#9a9a9a', lw=3.0, zorder=9))
    ey = hy + 0.022 * s
    for sx in (-1, 1):                                                                     # eyes, raised eyebrows
        circ(x + sx * 0.042 * s, ey, 0.014 * s, INK, z=11)
        if pose != 'point': ax.add_patch(Arc((x + sx * 0.042 * s, ey + 0.045 * s), 0.05 * s, 0.03 * s, theta1=20, theta2=160, color=INK, lw=0.6, zorder=11))
    if glasses:
        for sx in (-1, 1): circ(x + sx * 0.042 * s, ey, 0.034 * s, 'none', ec=INK, lw=0.6, z=11)
        seg([x - 0.008 * s, x + 0.008 * s], [ey, ey], c=INK, lw=0.6, z=11)
    if mustache: blob([(x - 0.05 * s, hy - 0.035 * s), (x, hy - 0.02 * s), (x + 0.05 * s, hy - 0.035 * s), (x, hy - 0.045 * s)], HAIR, z=11)
    if pose == 'point': seg([x - 0.03 * s, x + 0.03 * s], [hy - 0.065 * s, hy - 0.06 * s], c=INK, lw=0.6, z=11)   # composed mouth
    else: ell(x, hy - 0.065 * s, 0.038 * s, 0.052 * s, '#7a2e2e', z=11)              # open mouth
    if era == 1904 and hat:                                                                # bowler hat
        ell(x, hy + 0.09 * s, 0.31 * s, 0.04 * s, '#1b1b1b', z=12)
        blob([(x - 0.095 * s, hy + 0.095 * s), (x - 0.08 * s, hy + 0.19 * s), (x, hy + 0.215 * s), (x + 0.08 * s, hy + 0.19 * s), (x + 0.095 * s, hy + 0.095 * s)], '#1b1b1b', z=12)
        seg([x - 0.09 * s, x + 0.09 * s], [hy + 0.11 * s, hy + 0.11 * s], c='#5a4a3a', lw=1.0, z=13)

def bubble(x, y, w, h, tx, text):
    rbox(x, y, w, h, 'white', INK, lw=0.7, r=0.1, z=30)
    poly([(tx - 0.055, y + 0.012), (tx + 0.055, y + 0.012), (tx - 0.03, y - 0.15)], 'white', z=31)
    seg([tx - 0.055, tx - 0.03, tx + 0.055], [y, y - 0.15, y], c=INK, lw=0.7, z=31)
    T(x + w / 2, y + h / 2, text, fontsize=7.5, fontweight='bold', color=INK)

def bang(x, y, r=0): T(x, y, '!', fontsize=12, fontweight='bold', color=WOW, rotation=r)

G = 0.4
for x0, bg in ((0.04, BG_A), (3.58, BG_B)):
    rbox(x0, 0.1, 3.46, H - 0.32, bg, r=0.12, z=0)
    seg([x0 + 0.06, x0 + 3.4], [G, G], c='#cfc6b8' if bg == BG_A else '#c8d3de', lw=0.8, z=1)

# =================================== a: Berlin, 1904 ===================================
T(0.06, H - 0.1, 'a', fontsize=8, fontweight='bold', ha='left'); T(0.2, H - 0.1, 'Berlin, 1904', fontsize=7, ha='left')
# blackboard on an easel
for xa, xb in ((3.0, 2.95), (3.36, 3.41)): seg([xa, xb], [1.45, G + 0.01], c=WOOD, lw=1.6, z=2)
shadow(3.18, G, 0.55)
rbox(2.92, 1.4, 0.52, 0.5, BOARD, ec=WOOD, lw=1.6, r=0.02, z=3)
T(3.18, 1.66, '3 + 2 = ?', fontsize=7, color='#f4f4ee', fontweight='bold')
# Clever Hans, facing the board, one hoof raised
shadow(1.95, G, 1.2)
for pts, c in (([(1.56, 1.0), (1.53, 0.7), (1.55, G + 0.05)], HORSE_FAR), ([(2.13, 1.0), (2.14, 0.7), (2.15, G + 0.05)], HORSE_FAR)):
    taper(pts, 0.085, 0.06, c, z=3)
for pts in ([(1.68, 1.0), (1.66, 0.7), (1.67, G + 0.05)],):
    taper(pts, 0.09, 0.065, HORSE, z=4)
taper([(2.25, 1.0), (2.33, 0.8), (2.37, 0.62)], 0.09, 0.065, HORSE, z=4)                    # raised front leg
for hx, hy_, c in ((1.55, G + 0.03, HOOF), (2.15, G + 0.03, HOOF), (1.67, G + 0.03, HOOF)):
    rbox(hx - 0.04, hy_ - 0.03, 0.08, 0.05, c, r=0.012, z=5)
rbox(2.335, 0.57, 0.08, 0.05, HOOF, r=0.012, z=5)
blob([(1.42, 1.24), (1.6, 1.33), (1.85, 1.31), (2.08, 1.36), (2.22, 1.52), (2.36, 1.74), (2.48, 1.92), (2.56, 1.98), (2.7, 1.89),
      (2.84, 1.75), (2.9, 1.65), (2.87, 1.58), (2.77, 1.56), (2.64, 1.6), (2.56, 1.66), (2.5, 1.52), (2.42, 1.3), (2.36, 1.1),
      (2.28, 0.97), (2.1, 0.91), (1.85, 0.89), (1.62, 0.93), (1.48, 1.0), (1.4, 1.12)], HORSE, z=6)
poly([(2.5, 1.93), (2.5, 2.08), (2.6, 1.97)], HORSE, z=6)                                   # ear
poly([(2.53, 1.96), (2.53, 2.04), (2.58, 1.97)], HORSE_FAR, z=7)
stroke([(2.5, 1.95), (2.36, 1.75), (2.22, 1.53), (2.06, 1.38)], MANE, 4.2, z=7)             # mane
stroke([(2.5, 1.97), (2.6, 1.95), (2.67, 1.88)], MANE, 2.4, z=7)                             # forelock
taper([(1.43, 1.22), (1.33, 1.12), (1.3, 0.92), (1.34, 0.74)], 0.09, 0.03, MANE, z=5)        # tail
circ(2.65, 1.83, 0.022, INK, z=8); circ(2.656, 1.838, 0.007, 'white', z=9)                  # eye
circ(2.855, 1.64, 0.012, MANE, z=8)                                                         # nostril
for r in (0.07, 0.11, 0.15): ax.add_patch(Arc((2.375, 0.5), r * 2, r, theta1=200, theta2=340, color=GREY, lw=0.6, zorder=3))
T(2.6, 0.27, 'tap  tap  tap  tap  tap', fontsize=6, color=DGREY, style='italic')
# the questioner, and the amazed psychologists
person(1.05, G, '#6b4f3a', 1904, mustache=True, s=0.97)
person(0.3, G, '#3d4a5c', 1904, mustache=True)
person(0.67, G, '#4f5d4a', 1904, pose='cheeks', s=0.92)
bang(0.49, 2.05, 10); bang(0.86, 1.95, -8)
bubble(0.12, 2.32, 1.34, 0.38, 0.45, 'It can do arithmetic!')

# =================================== b: Anywhere, 2026 ===================================
X = 3.54
T(X + 0.06, H - 0.1, 'b', fontsize=8, fontweight='bold', ha='left'); T(X + 0.2, H - 0.1, 'Anywhere, 2026', fontsize=7, ha='left')
# screen on a stand with the outcome measure
seg([X + 3.15, X + 3.15], [1.4, G + 0.02], c='#5b6670', lw=1.8, z=2); ell(X + 3.15, G + 0.02, 0.3, 0.05, '#5b6670', z=2)
rbox(X + 2.84, 1.4, 0.62, 0.52, SCREEN, ec='#5b6670', lw=1.6, r=0.04, z=3)
T(X + 3.15, 1.77, 'How responsible\nis Sam? (1–9)', fontsize=6, color='white', linespacing=1.05)
circ(X + 3.15, 1.53, 0.075, WOW, z=5); T(X + 3.15, 1.53, '7', fontsize=8, color='white', fontweight='bold')
# the robot (an LLM)
RX = X + 2.06
shadow(RX, G, 0.62)
for sx in (-1, 1):
    taper([(RX + sx * 0.1, 0.98), (RX + sx * 0.11, 0.7), (RX + sx * 0.11, G + 0.06)], 0.075, 0.065, METAL_D, z=4)
    rbox(RX + sx * 0.11 - 0.07, G, 0.14, 0.06, '#7c8793', r=0.02, z=5)
rbox(RX - 0.28, 0.93, 0.56, 0.64, METAL, ec=METAL_D, lw=0.9, r=0.08, z=6)                     # body
rbox(RX - 0.22, 1.48, 0.12, 0.04, METAL_L, r=0.015, z=7)
circ(RX, 1.24, 0.2, SCREEN, ec=METAL_D, lw=0.9, z=7)                                          # chest window
L = {0: [(RX - 0.1, 1.24 + d) for d in (-0.09, 0, 0.09)], 1: [(RX, 1.24 + d) for d in (-0.105, -0.035, 0.035, 0.105)],
     2: [(RX + 0.1, 1.24 + d) for d in (-0.09, 0, 0.09)]}
for a_, b_ in ((0, 1), (1, 2)):
    for p in L[a_]:
        for q in L[b_]: seg([p[0], q[0]], [p[1], q[1]], c='#4f6b86', lw=0.3, z=8)
for k_, layer in L.items():
    for j, p in enumerate(layer): circ(p[0], p[1], 0.019, '#ffd29a' if (k_ == 1 and j == 2) else '#8fb0d6', z=9)
for sx in (-1, 1):                                                                            # arms
    if sx > 0:
        taper([(RX + 0.26, 1.45), (RX + 0.5, 1.56), (RX + 0.72, 1.55)], 0.075, 0.06, METAL_D, z=5)
        circ(RX + 0.74, 1.55, 0.04, '#7c8793', z=6)
    else:
        taper([(RX - 0.26, 1.45), (RX - 0.37, 1.22), (RX - 0.35, 1.02)], 0.075, 0.06, METAL_D, z=5)
        circ(RX - 0.35, 1.0, 0.04, '#7c8793', z=6)
rbox(RX - 0.05, 1.56, 0.1, 0.06, METAL_D, r=0.01, z=5)                                      # neck
rbox(RX - 0.22, 1.6, 0.44, 0.36, METAL, ec=METAL_D, lw=0.9, r=0.09, z=7)                     # head
for sx in (-1, 1): rbox(RX + sx * 0.235 - 0.03, 1.72, 0.06, 0.12, METAL_D, r=0.02, z=6)       # ear discs
rbox(RX - 0.165, 1.655, 0.33, 0.25, SCREEN, r=0.06, z=8)
for sx in (-1, 1): ell(RX + sx * 0.068, 1.79, 0.075, 0.1, GLOW, z=9); ell(RX + sx * 0.06, 1.81, 0.022, 0.03, 'white', z=10)
ax.add_patch(Arc((RX, 1.715), 0.13, 0.06, theta1=200, theta2=340, color=GLOW, lw=1.0, zorder=9))
seg([RX, RX], [1.96, 2.08], c=METAL_D, lw=1.2, z=6); circ(RX, 2.1, 0.032, WOW, z=7)
# the amazed psychologists
person(X + 0.32, G, '#2f6fad', 2026, glasses=True, hat=False, hair='short')
person(X + 0.7, G, '#d9822b', 2026, pose='cheeks', s=0.92, hat=False, hair='bun')
person(X + 1.08, G, '#7a8b99', 2026, s=0.96, hat=False, hair='grey')
bang(X + 0.51, 2.05, 10); bang(X + 0.89, 1.95, -8)
bubble(X + 0.12, 2.32, 1.66, 0.38, X + 0.47, 'It responds like a human!')

M.VALS['fig_cleverhans'] = {'kind': 'illustration (no data)'}
M.save(fig, 'fig_cleverhans')
pv = M.OUT / 'plotted_values.json'
allv = json.load(open(pv, encoding='utf-8')) if pv.exists() else {}
allv.update(json.loads(json.dumps(M.VALS, default=str))); json.dump(allv, open(pv, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
