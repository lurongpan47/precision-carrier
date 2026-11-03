# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import wmp as m

R = json.load(open('results.json'))
A = np.load('arrays.npz')
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', violet='#6250d6', red='#e34948')
plt.rcParams.update({'font.family': ['Liberation Sans', 'DejaVu Sans'], 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.titlecolor': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'lines.linewidth': 1.8, 'mathtext.fontset': 'dejavusans', 'legend.frameon': False, 'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                     'savefig.facecolor': SURF, 'axes.titlelocation': 'left'})
LV = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']
BLUES = LinearSegmentedColormap.from_list('b', ['#f3f7fc', '#a9c9ee', '#2a78d6', '#123a6b'])
P = m.Params()

# ---------- Fig 1 ----------
fig = plt.figure(figsize=(7.4, 2.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=.34)
ax = fig.add_subplot(gs[0])
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('a  Model')
ys = np.linspace(.7, 8.8, 7)
for i, y in enumerate(ys):
    ax.add_patch(mp.FancyBboxPatch((.4, y - .4), 2.1, .8, boxstyle='round,pad=0.03,rounding_size=.12', fc='#eef4fb', ec=C['blue'], lw=.8))
    ax.text(1.45, y, LV[i], ha='center', va='center', fontsize=7.5, color=INK)
    if i < 6:
        ax.plot([1.45, 1.45], [y + .42, ys[i + 1] - .42], color=MUTED, lw=1.1)
ax.text(.0, 4.75, r'$\kappa$', fontsize=8.5, color=INK2, ha='center')
ax.add_patch(mp.FancyBboxPatch((5.3, 6.0), 4.4, 1.7, boxstyle='round,pad=0.03,rounding_size=.15', fc='#f2f0fb', ec=C['violet'], lw=.9))
ax.text(7.45, 6.85, 'precision $g$', ha='center', va='center', fontsize=8, color=INK)
ax.add_patch(mp.FancyBboxPatch((5.3, 2.0), 4.4, 1.7, boxstyle='round,pad=0.03,rounding_size=.15', fc='#fdf0e9', ec=C['orange'], lw=.9))
ax.text(7.45, 2.85, 'mismatch $d^2$', ha='center', va='center', fontsize=8, color=INK)
ax.annotate('', xy=(2.65, 6.7), xytext=(5.2, 6.85), arrowprops=dict(arrowstyle='->', color=C['violet'], lw=1.1))
ax.text(4.05, 7.25, 'gain', fontsize=7.2, color=INK2, ha='center')
ax.annotate('', xy=(5.2, 6.25), xytext=(2.65, 4.2), arrowprops=dict(arrowstyle='->', color=C['orange'], lw=1.1))
ax.text(5.6, 4.6, '$W=\\Sigma w_ix_i$', fontsize=7, color=INK2, ha='center')
ax.annotate('', xy=(7.45, 5.95), xytext=(7.45, 3.75), arrowprops=dict(arrowstyle='->', color=C['orange'], lw=1.1))
ax.text(4.1, .75, 'only committed\nbeliefs pay error', fontsize=7, color=INK2, ha='left', va='center')
ax2 = fig.add_subplot(gs[1])
xs = np.linspace(0, .999, 800)
for bj, col, lab in [(3.2, C['blue'], '$gJ$ = 3.2: flat only'), (5.5, C['orange'], '$gJ$ = 5.5: both'), (8., C['aqua'], '$gJ$ = 8: flat unstable')]:
    f = m.F_single(xs, 7, bj)
    ax2.plot(xs, f - f[0], color=col, label=lab)
ax2.set_xlabel('commitment $x$')
ax2.set_ylabel('$F(x)-F(0)$')
ax2.set_ylim(-1.6, 1.0)
ax2.legend(fontsize=7, loc='lower left')
ax2.set_title('b  Free energy ($q$ = 7)')
ax3 = fig.add_subplot(gs[2])
b = A['mc_bJ']
ax3.plot(b, A['mf_down'], color=C['blue'], label='MF, falling')
ax3.plot(b, A['mf_up'], color=C['orange'], label='MF, rising')
ax3.plot(b, A['mc_down'], 'o', ms=3.5, mfc=SURF, mec=C['blue'], mew=.8, label='MC, falling')
ax3.plot(b, A['mc_up'], 's', ms=3.5, mfc=SURF, mec=C['orange'], mew=.8, label='MC, rising')
ax3.set_xlabel('$gJ$')
ax3.set_ylabel('$x$')
ax3.legend(fontsize=6.6, loc='center', bbox_to_anchor=(.5, .42), handlelength=1.2, borderaxespad=0)
ax3.set_title('c  Hysteresis')
fig.savefig('Fig1.png')

# ---------- Fig 2 ----------
t = A['dy_t']
st = R['dying']['stage_t']
fig, axs = plt.subplots(3, 1, figsize=(6.6, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.25, 1.25], hspace=.32))
a0 = axs[0]
a0.imshow(A['dy_X'][:, :].T, aspect='auto', origin='lower', cmap=BLUES, vmin=0, vmax=1, extent=[t[0], t[-1], -.5, 6.5], interpolation='nearest')
a0.set_yticks(range(7))
a0.set_yticklabels(LV)
a0.grid(False)
a0.set_title('a  Commitment $x_i$ (darker = more determinate)')
axs[1].plot(t, A['dy_gflat'], color=MUTED, lw=1.2, label='flat-state precision $a(t)/b$')
axs[1].plot(t, A['dy_G'], color=C['blue'], label='precision $g$, uniform weights')
axs[1].plot(A['dyl_t'], A['dyl_G'], color=C['orange'], lw=1.3, label=r'precision $g$, weights tilted to L7 ($\lambda$ = 2)')
axs[1].set_ylabel('$g$')
axs[1].set_ylim(0, 1.3)
axs[1].legend(fontsize=7.5, loc='upper right')
axs[1].set_title('b  Precision falls, and rebounds to $a/b$ once the last level collapses')
Rv = A['dy_R'].copy()
Rv[~np.isfinite(Rv)] = np.nan
axs[2].plot(t, Rv, color=C['blue'])
Rl = A['dyl_R'].copy()
Rl[~np.isfinite(Rl)] = np.nan
axs[2].plot(A['dyl_t'], Rl, color=C['orange'], lw=1.1)
axs[2].set_yscale('log')
axs[2].set_ylim(.08, 6)
axs[2].set_yticks([.1, .3, 1, 3])
axs[2].set_yticklabels(['0.1', '0.3', '1', '3'])
axs[2].set_ylabel('readiness $R$ (nat)')
axs[2].set_xlabel('time (arbitrary units)')
axs[2].set_title('c  Readiness: a critical spike at each collapse, and no recovery after the last (baseline)')
for a_ in axs:
    for s_ in st:
        a_.axvline(s_, color=INK2, lw=.5, ls=(0, (2, 2)))
axs[0].set_xlim(0, 300)
fig.savefig('Fig2.png')

# ---------- Fig 3 ----------
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.55), gridspec_kw=dict(width_ratios=[1.15, 1, 1.15], wspace=.36))
l, r_ = R['ramp']['loss'], R['ramp']['ret']
for i in range(7):
    ax[0].plot([l[i], r_[i]], [i, i], color=GRID, lw=2.6, zorder=1)
ax[0].scatter(l, range(7), color=C['blue'], s=26, zorder=3, label='loss ($g$ falling)')
ax[0].scatter(r_, range(7), color=C['orange'], s=26, zorder=3, label='return ($g$ rising)')
ax[0].set_yticks(range(7))
ax[0].set_yticklabels(LV)
ax[0].set_xlabel('critical precision $g$')
ax[0].legend(fontsize=6.3, loc='upper right', handletextpad=.1, borderaxespad=0)
ax[0].grid(axis='y', visible=False)
ax[0].set_title('a  Thresholds')
la = R['loop_area']
ax[1].plot([d['T'] for d in la], [d['area'] for d in la], '-o', ms=3.5, color=C['violet'])
ax[1].set_xscale('log')
ax[1].set_ylim(0, None)
ax[1].set_xlabel('sweep duration')
ax[1].set_ylabel('loop area')
ax[1].set_title('b  Loop persists')
tm = A['med_t']
sel = (tm > 270) & (tm < 400)
ax[2].imshow(A['med_X'][sel].T, aspect='auto', origin='lower', cmap=BLUES, vmin=0, vmax=1, extent=[tm[sel][0], tm[sel][-1], -.5, 6.5], interpolation='nearest')
ax[2].set_yticks(range(7))
ax[2].set_yticklabels(LV)
ax[2].grid(False)
ax[2].set_xlabel('time (arbitrary units)')
ax[2].set_title('c  Return as a front')
fig.savefig('Fig3.png')

# ---------- Fig 4 ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=.42))
Z = A['rb']
im = ax[0].imshow(Z, origin='lower', aspect='auto', cmap=BLUES, vmin=1, vmax=np.nanmax(Z))
ax[0].set_xticks(range(len(A['rb_Ds'])))
ax[0].set_xticklabels([f'{v:g}' for v in A['rb_Ds']])
ax[0].set_yticks(range(len(A['rb_lams'])))
ax[0].set_yticklabels([f'{v:g}' for v in A['rb_lams']])
ax[0].grid(False)
ax[0].set_xlabel('mismatch slope $D$')
ax[0].set_ylabel('weight tilt towards L7, $\\lambda$')
ax[0].set_title('a  Rebound $g_{max}/g_{min}$')
for i in range(Z.shape[0]):
    for k in range(Z.shape[1]):
        if np.isfinite(Z[i, k]):
            ax[0].text(k, i, f'{Z[i, k]:.2f}', ha='center', va='center', fontsize=6.6, color=SURF if Z[i, k] > .55 * (np.nanmax(Z) + 1) else INK)
fig.colorbar(im, ax=ax[0], fraction=.046, pad=.02)
rho, reb = A['sw_rho'], A['sw_reb']
ax[1].scatter(rho, reb, s=7, color=C['blue'], alpha=.55, linewidths=0)
ax[1].axvline(1, color=C['orange'], lw=1.1)
ax[1].axhline(1, color=MUTED, lw=.8)
ax[1].set_xscale('log')
ax[1].set_xlabel(r'$\rho_c=\omega d^2a/b$ at the last collapse')
ax[1].set_ylabel('rebound')
ax[1].set_title('b  The criterion across random parameters')
fig.savefig('Fig4.png')

# ---------- Fig 5 ----------
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw=dict(width_ratios=[1.45, 1, 1], wspace=.4))
Ry = A['dy_R'].copy()
Ry[~np.isfinite(Ry)] = np.nan
ax[0].plot(A['dy_E'], Ry, color=C['orange'], label='failing support')
tm = A['med_t']
sel = tm <= 300
Rm = A['med_R'].copy()
Rm[~np.isfinite(Rm)] = np.nan
ax[0].plot(A['med_E'][sel], Rm[sel], color=C['violet'], label='prior relaxation')
ax[0].set_yscale('log')
ax[0].set_ylim(.09, 16)
ax[0].set_yticks([.1, .3, 1, 3, 10])
ax[0].set_yticklabels(['0.1', '0.3', '1', '3', '10'])
ax[0].set_xlabel(r'emptiness $\varepsilon$')
ax[0].set_ylabel('readiness $R$ (nat)')
S = {s['key']: s for s in R['states']}
for key, lab, col, off in [('wake', 'waking', C['aqua'], (6, -2)), ('sleep', 'deep sleep-like', C['blue'], (5, -8)), ('nadir', 'last stage', C['orange'], (-38, -3)),
                           ('dycl', 'after collapse', C['orange'], (5, 0)), ('medcl', 'hold', C['violet'], (5, -2))]:
    s = S[key]
    ax[0].scatter([s['eps']], [s['R']], s=22, color=col, zorder=6, edgecolor=SURF, linewidth=.6)
    ax[0].annotate(lab, (s['eps'], s['R']), xytext=off, textcoords='offset points', fontsize=6.8, color=INK, bbox=dict(fc=SURF, ec='none', pad=.3, alpha=.8), zorder=7)
ax[0].set_xlim(-.03, 1.62)
ax[0].set_xticks([0, .25, .5, .75, 1])
ax[0].legend(fontsize=6.8, loc='lower left')
ax[0].set_title('a  Readiness and emptiness')
tv = A['ev_t']
post = tv > .5
for a_, kd, ttl in [(ax[1], 'evidence', 'b  Evidence pulse'), (ax[2], 'direct', 'c  Direct pulse')]:
    for key, lab, col in [('sleep', 'deep sleep-like', C['blue']), ('dycl', 'after collapse', C['orange']), ('medcl', 'hold', C['violet']), ('medcrit', 'near instability', C['red'])]:
        y_ = np.abs(A[f'ev_{kd}_' + key][post]).sum(1)
        y_[y_ < 1e-4] = np.nan
        a_.plot(tv[post] - .5, y_, color=col, label=lab, lw=1.4)
    a_.set_yscale('log')
    a_.set_xlabel('time after pulse (s)')
    a_.set_title(ttl)
ax[1].set_ylabel(r'evoked response, $\Sigma_i|\delta u_i|$')
ax[2].legend(fontsize=6.3, loc='lower left')
fig.savefig('Fig5.png')

# ---------- Fig 6 ----------
gg, jj, pa, pb, Rp = A['ph_g'], A['ph_js'], A['ph_a'], A['ph_b'], A['ph_R']
cls = np.zeros_like(pa)
cls[(pa >= .1) & (pb < .1)] = 1
cls[(pa >= .1) & (pb >= .1) & (np.abs(pa - pb) >= .05)] = 3
cls[(pa >= .1) & (pb >= .1) & (np.abs(pa - pb) < .05)] = 2
fig = plt.figure(figsize=(7.4, 3.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], wspace=.3, hspace=.62)
ax = fig.add_subplot(gs[:, 0])
ax.pcolormesh(gg, jj, cls, cmap=ListedColormap(['#eef4fb', '#fbe9d9', '#dff3ea', '#fdf3d8']), vmin=-.5, vmax=3.5, shading='nearest', rasterized=True)
cs = ax.contour(gg, jj, np.ma.masked_invalid(np.where(np.isfinite(Rp), Rp, np.nan)), levels=[.3, 1, 2], colors=C['blue'], linewidths=.9)
ax.clabel(cs, fmt='R=%g', fontsize=6.8)
ax.grid(False)
ax.set_xlabel('precision $g$')
ax.set_ylabel('prior precision $J_s$')
for key, lab, col, off in [('wake', 'waking / dreaming-like', C['aqua'], (6, 4)), ('sleep', 'deep sleep-like', C['blue'], (6, 4)), ('medcl', 'hold', C['violet'], (6, -9)), ('medcrit', 'near instability', C['red'], (6, 4))]:
    s = S[key]
    ax.scatter([s['g']], [s['Js']], s=26, color=col, edgecolor=SURF, linewidth=.6, zorder=5)
    ax.annotate(lab, (s['g'], s['Js']), xytext=off, textcoords='offset points', fontsize=7.2, color=INK, bbox=dict(fc=SURF, ec='none', pad=.4, alpha=.75))
ax.set_xlim(gg[0], gg[-1])
ax.set_ylim(jj[0], jj[-1])
hs = [mp.Patch(color='#eef4fb', label='flat only'), mp.Patch(color='#fbe9d9', label='flat and contentful coexist'), mp.Patch(color='#fdf3d8', label='partly flat'), mp.Patch(color='#dff3ea', label='contentful only')]
ax.legend(handles=hs, fontsize=6.8, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4, handlelength=1, columnspacing=.8)
tr = R['tradeoff']
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot([d['Js'] for d in tr], [d['R'] for d in tr], '-o', ms=3.2, color=C['violet'])
ax1.set_ylabel('$R$ (nat)')
ax1.set_title('b  Readiness rises...')
ax1.set_xticklabels([])
ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
trb = [d for d in tr if d['dF'] > 0]
ax2.plot([d['Js'] for d in trb], [d['dF'] for d in trb], '-o', ms=3.2, color=C['orange'])
ax2.set_xlabel('prior precision $J_s$ (at $g$ = 1.2)')
ax2.set_ylabel(r'barrier $\Delta F$')
ax2.set_title('c  ...as stability falls')
fig.savefig('Fig6.png')

# ---------- Fig 7 ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=.3))
for r, lab, col, mk in [(R['dying']['stage_t'], 'global precision withdrawal', C['blue'], 'o'), (R['route_bottomup'], 'relaxation, bottom-up', C['aqua'], 's'), (R['route_topdown'], 'relaxation, top-down', C['orange'], '^')]:
    ax[0].plot(r, range(7), '-' + mk, ms=3.6, color=col, label=lab, lw=1.3)
ax[0].set_yticks(range(7))
ax[0].set_yticklabels(LV)
ax[0].set_xlabel('time of loss (arbitrary units)')
ax[0].set_xlim(50, 400)
ax[0].set_xticks([100, 150, 200])
ax[0].legend(fontsize=7, loc='center right')
ax[0].set_title('a  Imposed orders')
f_ = A['spec_f']
for key, lab, col in [('sleep', 'deep sleep-like', C['blue']), ('dycl', 'after collapse', C['orange']), ('medcl', 'hold', C['violet']), ('medcrit', 'near instability', C['red'])]:
    ax[1].loglog(f_, A['spec_' + key], color=col, label=lab)
ax[1].axvspan(1, 4, color='#f1f0ed', zorder=0)
ax[1].set_xlabel('frequency (Hz)')
ax[1].set_ylabel('fluctuation power')
ax[1].legend(fontsize=7, loc='lower left')
ax[1].set_title('b  Fluctuation spectra (grey: 1\u20134 Hz)')
fig.savefig('Fig7.png')
print('figs ok')
