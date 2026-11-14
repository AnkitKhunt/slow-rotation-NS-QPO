"""
plot_style.py -- common figure style for the HT QPO paper.

Computer Modern fonts (bundled with Matplotlib, no LaTeX needed), bold tick
labels, labels in math mode, thick axes, major and minor ticks inward on all
four sides.

    import plot_style as ps
    ps.use()
    fig, ax = plt.subplots()
    ...
    ax.set_xlabel(ps.lab(r'\\nu_{\\phi}', 'Hz'))
    ps.finish(ax)
"""
import logging

import numpy as np

import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator

# EOS order, labels, colours (colour-blind-safe) and line styles
EOS_ORDER = ['eosAU', 'G3', 'FSUGarnet', 'IOPB']
EOS_LABEL = {'eosAU': 'WFF1', 'G3': 'G3', 'FSUGarnet': 'FSUGarnet',
             'IOPB': 'IOPB-I'}
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
LS = ['-', '--', '-.', ':']
EOS_COL = dict(zip(EOS_ORDER, COL))
EOS_LS = dict(zip(EOS_ORDER, LS))
GREY = '#555555'

# bold math symbols used in labels
S = {
    'nuphi': r'\boldsymbol{\nu}_{\boldsymbol{\phi}}',
    'nuth': r'\boldsymbol{\nu}_{\boldsymbol{\theta}}',
    'nur': r'\boldsymbol{\nu}_{\boldsymbol{r}}',
    'nuper': r'\boldsymbol{\nu}_{\mathbf{per}}',
    'nunod': r'\boldsymbol{\nu}_{\mathbf{nod}}',
    'nuLT': r'\boldsymbol{\nu}_{\mathbf{LT}}',
    'nuQ': r'\boldsymbol{\nu}_{\boldsymbol{Q}}',
    'nuU': r'\boldsymbol{\nu}_{\mathbf{U}}',
    'nuL': r'\boldsymbol{\nu}_{\mathbf{L}}',
    'nuLF': r'\boldsymbol{\nu}_{\mathbf{LF}}',
    'nuHBO': r'\boldsymbol{\nu}_{\mathbf{HBO}}',
    'nus': r'\boldsymbol{\nu}_{\boldsymbol{s}}',
    'M': r'\boldsymbol{M}', 'Msun': r'\boldsymbol{M}_{\boldsymbol{\odot}}',
    'R': r'\boldsymbol{R}', 'r': r'\boldsymbol{r}', 'j': r'\boldsymbol{j}',
    'Qbar': r'\bar{\boldsymbol{Q}}', 'I': r'\boldsymbol{I}',
    't': r'\boldsymbol{t}', 'x': r'\boldsymbol{x}', 'y': r'\boldsymbol{y}',
    'z': r'\boldsymbol{z}', 'f': r'\boldsymbol{f}',
}

RC = {
    'font.family': 'serif',
    'font.serif': ['cmb10'],            # bold Computer Modern for text/ticks
    'mathtext.fontset': 'cm',
    'mathtext.default': 'bf',
    'axes.unicode_minus': False,
    'axes.formatter.use_mathtext': False,
    'axes.linewidth': 1.3,
    'axes.labelsize': 12,
    'axes.titlesize': 11,
    'axes.labelpad': 4.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.major.size': 5.5, 'ytick.major.size': 5.5,
    'xtick.minor.size': 3.0, 'ytick.minor.size': 3.0,
    'xtick.major.width': 1.3, 'ytick.major.width': 1.3,
    'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
    'xtick.major.pad': 4.0, 'ytick.major.pad': 4.0,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'legend.fontsize': 8.5, 'legend.frameon': False,
    'legend.handlelength': 2.4,
    'lines.linewidth': 1.7,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42,
}


def use():
    """Apply the paper style.  Needs Matplotlib >= 3.8 (\\boldsymbol)."""
    v = tuple(int(''.join(c for c in x if c.isdigit()) or 0)
              for x in mpl.__version__.split('.')[:2])
    if v < (3, 8):
        raise RuntimeError(
            f'Matplotlib {mpl.__version__} is too old for the figure labels '
            '(\\boldsymbol needs 3.8 or newer).  Run:  '
            'pip install -U "matplotlib>=3.8"')
    logging.getLogger('fontTools').setLevel(logging.ERROR)   # quiet PDF font subsetting
    mpl.rcParams.update(RC)


def lab(sym, unit=None):
    """Axis label: bold math symbol plus an optional unit in brackets."""
    s = r'$' + sym + r'$'
    return s if unit is None else s + r' [' + unit + r']'


def _logfmt(v, pos):
    if v <= 0:
        return ''
    e = np.log10(v)
    if abs(e - round(e)) > 1e-6:
        return ''
    e = int(round(e))
    if e == 0:
        return r'$\mathbf{1}$'
    if e == 1:
        return r'$\mathbf{10}$'
    return r'$\mathbf{10^{%d}}$' % e


def finish(ax):
    """Bold log tick labels, inward ticks on all sides."""
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() == 'log':
            axis.set_major_locator(LogLocator(base=10, numticks=12))
            axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10),
                                              numticks=12))
            axis.set_major_formatter(FuncFormatter(_logfmt))
            axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='both', direction='in', top=True, right=True)
    return ax


def panel_tag(ax, s, x=0.04, y=0.94, **kw):
    """Fixed-position panel label (kept for old scripts)."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, va='top',
            ha='left', **kw)


# ---------------------------------------------------------------------
# Panel labels (a), (b), ... inside the axes, placed where they do not
# touch curves, data points, error bars, legends or other text.
# ---------------------------------------------------------------------
# candidate positions: (x, y) in axes fraction, ha, va, offset sign
_SPOTS = [((0, 1), 'left', 'top'), ((1, 1), 'right', 'top'),
          ((0, 0), 'left', 'bottom'), ((1, 0), 'right', 'bottom'),
          ((0.5, 1), 'center', 'top'), ((0.5, 0), 'center', 'bottom'),
          ((0, 0.5), 'left', 'center'), ((1, 0.5), 'right', 'center')]


def _dense(xy, step=1.5):
    """Points along a polyline in display units, about `step` px apart."""
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if len(xy) < 2:
        return xy
    seg = np.diff(xy, axis=0)
    n = np.maximum(1, np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / step)).astype(int)
    out = [xy[:-1] + seg * (k / n)[:, None]
           for k in range(int(n.max()))]
    pts = np.concatenate([o[k < n] for k, o in enumerate(out)] + [xy[-1:]])
    return pts


def _obstacles(ax, renderer, skip):
    """Display-space points of everything drawn in ax, and text boxes."""
    pts = []
    for ln in ax.lines:
        if not ln.get_visible() or ln in skip:
            continue
        xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        if ln.get_linestyle() not in ('None', '', ' ', 'none'):
            pts.append(_dense(xy))
        else:
            pts.append(xy[np.all(np.isfinite(xy), axis=1)])
    for c in ax.collections:
        if not c.get_visible():
            continue
        if hasattr(c, 'get_segments'):                    # error bars, lines
            tr = c.get_transform()
            for sgm in c.get_segments():
                if len(sgm):
                    pts.append(_dense(tr.transform(np.asarray(sgm, float))))
        off = c.get_offsets()
        if len(off) and not hasattr(c, 'get_segments'):  # scatter markers
            pts.append(c.get_offset_transform().transform(off))
    boxes = [t.get_window_extent(renderer) for t in ax.texts
             if t.get_visible() and t not in skip and t.get_text().strip()]
    leg = ax.get_legend()
    if leg is not None and leg.get_visible():
        boxes.append(leg.get_window_extent(renderer))
    pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    return pts, boxes


def _score(bb, pts, boxes, pad):
    x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
    n = int(np.count_nonzero((pts[:, 0] > x0) & (pts[:, 0] < x1) &
                             (pts[:, 1] > y0) & (pts[:, 1] < y1)))
    for b in boxes:
        if b.x0 < x1 and b.x1 > x0 and b.y0 < y1 and b.y1 > y0:
            n += 10 ** 6
    return n


_WHERE = {'any': range(8), 'corners': range(4), 'top': (0, 1)}


def _best_spot(ax, txt, renderer, pad_pt, where='any', clear_pt=5.0):
    pts, boxes = _obstacles(ax, renderer, skip=[txt])
    clear = clear_pt * ax.figure.dpi / 72.0          # points -> pixels
    best = None
    for k in _WHERE[where]:
        (fx, fy), ha, va = _SPOTS[k]
        dx = pad_pt if fx == 0 else (-pad_pt if fx == 1 else 0)
        dy = pad_pt if fy == 0 else (-pad_pt if fy == 1 else 0)
        txt.xy = (fx, fy)
        txt.set_position((dx, dy))
        txt.set_ha(ha); txt.set_va(va)
        sc = _score(txt.get_window_extent(renderer), pts, boxes, clear)
        if best is None or sc < best[0]:
            best = (sc, (fx, fy), (dx, dy), ha, va)
        if sc == 0:
            break                       # first free spot in preference order
    return best


def _grow_top(ax, frac):
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == 'log':
        ax.set_ylim(lo, hi * (hi / lo) ** frac)
    else:
        ax.set_ylim(lo, hi + frac * (hi - lo))


def label_panels(fig, axes, labels=None, fontsize=11, pad_pt=6.5,
                 make_room=True, max_grow=4, grow=0.08, where='any',
                 clear_pt=5.0):
    """
    Put (a), (b), ... inside each axes where they overlap nothing.

    Call after all plotting and after tight_layout, just before saving.
    For every axes the label is tried at eight positions (corners first,
    then edge midpoints); the first position free of curves, data points,
    error bars, legends and text is used.  If no position is free and
    make_room is True, the upper y limit is raised by `grow` (fraction of
    the range, or of the decades on a log axis) up to `max_grow` times.
    The label sits pad_pt points from the spines, clear of the inward
    ticks, and keeps clear_pt points of empty space around it.  where='corners' or 'top' restricts the positions (use 'top'
    for oscillating time series, where a label between peaks looks odd).
    """
    axes = list(np.ravel(axes))
    labels = labels or [f'({chr(97 + i)})' for i in range(len(axes))]
    out = []
    for ax, s in zip(axes, labels):
        txt = ax.annotate(s, xy=(0, 1), xycoords='axes fraction',
                          xytext=(pad_pt, -pad_pt), textcoords='offset points',
                          ha='left', va='top', fontsize=fontsize, zorder=10,
                          annotation_clip=False)
        for it in range(max_grow + 1):
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            best = _best_spot(ax, txt, r, pad_pt, where, clear_pt)
            if best[0] == 0 or not make_room or it == max_grow:
                break
            _grow_top(ax, grow)
        sc, xy, off, ha, va = best
        txt.xy = xy
        txt.set_position(off)
        txt.set_ha(ha); txt.set_va(va)
        if sc:
            print(f'  label {s}: no free spot, {sc} points underneath')
        out.append(txt)
    return out
