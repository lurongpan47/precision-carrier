# -*- coding: utf-8 -*-
"""Reproduces every number in the manuscript. Usage: python3 run_v2.py  (writes results.json, arrays.npz)."""
import json, time, itertools
import numpy as np
from multiprocessing import Pool
from scipy import stats
from scipy.optimize import brentq, curve_fit
import wmp as m

A0, E0, D0, T0 = 1.2, 1.25, 8.0, 300.0
HW = np.array([.5, 0, 0, 0, 0, 0, 0.])
EPS = 2e-3 * np.ones(7)
LOST, BACK = .15, .5
R, A = {}, {}
t00 = time.time()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and (np.isinf(o) or np.isnan(o)):
        return str(o)
    return o


def save(tag):
    json.dump(clean(R), open('results.json', 'w'), indent=1)
    np.savez_compressed('arrays.npz', **A)
    print(tag, round(time.time() - t00), flush=True)


def g_wake(P, a=A0, e=E0):
    return (a + .5) / (P.b + .5 * P.omega * e)


def wake_state(P, a=A0, e=E0, h=HW + EPS):
    return m.relax(np.ones(P.N) * .95, g_wake(P, a, e), P, h, e)


def dying(P, T=T0, D=D0, a0=A0, e0=E0, dt=.02, sub=1, seeds=1, noise=0., seed=0, g_fn=None, hw=HW):
    g0 = g_wake(P, a0, e0)
    x0 = m.relax(np.ones(P.N) * .95, g0, P, hw + EPS, e0)
    return m.simulate(P, T, dt, lambda t: a0 * max(1 - t / T, 0.), lambda t: e0 + D * t / T,
                      lambda t: EPS + hw * max(1 - t / (.2 * T), 0.), x0=x0, g0=g0, noise=noise, seeds=seeds, seed=seed,
                      sub=sub, g_fn=g_fn)


def stage_times(t, X):
    return [float(t[np.argmax(X[:, i] < LOST)]) if np.any(X[:, i] < LOST) else None for i in range(X.shape[1])]


def rebound_metrics(t, X, G, P=None, a0=A0, e0=E0, D=D0, T=T0):
    st = stage_times(t, X)
    if None in st:
        return {'complete': False, 'stage_t': st}
    o = np.argsort(st)
    last, prev = o[-1], o[-2]
    tc, tp = st[last], st[prev]
    pre = (t > tp) & (t <= tc)
    if pre.sum() < 2:
        pre = (t > tc - 10) & (t <= tc)
    post = t > tc
    i = np.argmin(np.where(pre, G, np.inf))
    res = {'complete': True, 'stage_t': st, 'g_min': float(G[i]), 't_gmin': float(t[i]), 'g_max': float(G[post].max()),
           't_gmax': float(t[post][np.argmax(G[post])]), 'rebound': float(G[post].max() / G[i]), 't_c': float(tc),
           'order': [int(v) for v in o]}
    if P is not None:
        res['rho_c'] = float(P.omega * (e0 + D * tc / T) * a0 * max(1 - tc / T, 1e-9) / P.b)
    return res


def ramp(P, d2=E0, dg=.005, T=4., gmax=1.8, gmin=.02, dt=.01, Js=1.0):
    gs = np.arange(gmax, gmin, -dg)
    x = np.ones(P.N) * .95
    loss, ret = [None] * P.N, [None] * P.N
    Dn, Up = [], []
    for g in gs:
        x = m.relax(x, g, P, EPS[:P.N], d2, Js, T=T, dt=dt)
        Dn.append(x.copy())
        for i in range(P.N):
            if loss[i] is None and x[i] < LOST:
                loss[i] = float(g)
    for g in gs[::-1]:
        x = m.relax(x, g, P, EPS[:P.N], d2, Js, T=T, dt=dt)
        Up.append(x.copy())
        for i in range(P.N):
            if ret[i] is None and x[i] > BACK:
                ret[i] = float(g)
    return gs, np.array(Dn), np.array(Up), loss, ret


def kendall_to_canonical(st):
    return stats.kendalltau(np.arange(len(st)), st).statistic


# ---------- S0 Analytic and numerical checks ----------
def coexist_numeric(q):
    def f(bj):
        s, xs = m.lower_spinodal(q)
        grid = np.linspace(xs, 1 - 1e-7, 200001)
        return m.F_single(grid, q, bj).min() - m.F_single(0., q, bj)
    return brentq(f, m.lower_spinodal(q)[0] + 1e-6, q - 1e-6, xtol=1e-6)


R['analytic'] = [{'q': q, 'lower': m.lower_spinodal(q)[0], 'x_lower': m.lower_spinodal(q)[1], 'coex_formula': m.coexistence(q),
                  'coex_numeric': coexist_numeric(q), 'upper': q} for q in [3, 7, 33, 40]]


def fd_gain(q, gJ, g, h0=1e-5, dh=1e-6):
    """Independent finite-difference gain of an isolated level (solve u = gJ x(u) + q/(q-1) g h by bisection)."""
    def fp(h):
        # lowest root on the flat branch: scan upward from u = 0 for the first sign change, then bisect
        fn = lambda u: gJ * m.x_of_u(u, q) + q / (q - 1) * g * h - u
        lo, hi = 0., None
        for u in np.linspace(0, 2, 20001)[1:]:
            if fn(u) < 0:
                hi = u
                break
            lo = u
        for _ in range(200):
            mid = (lo + hi) / 2
            if fn(mid) > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    return (fp(h0 + dh) - fp(h0)) / dh


R['chi_check'] = [{'bJ': b, 'fd': fd_gain(7., b, b / 10), 'analytic': (7 / 6) * (b / 10) / (1 - b / 7)} for b in [1, 3, 5, 6.5]]

P = m.Params()
# Lyapunov check at fixed g: F must not increase under the belief dynamics
rng = np.random.default_rng(3)
viol, maxinc = 0, 0.
for trial in range(50):
    x = rng.uniform(.0, .99, 7)
    g = rng.uniform(.1, 1.8)
    js = rng.uniform(.2, 1)
    d2 = rng.uniform(0, 8)
    h = EPS + HW * rng.uniform(0, 1)
    u = m.u_of_x(x, P.q)
    prev = m.free_energy(x, g, P, h, d2, js)
    for k in range(3000):
        x = np.clip(m.x_of_u(u, P.q), 0, 1)
        u = u + .002 / P.tau * (m.u_target(x, g, P, h, d2, js) - u)
        if k % 30 == 0:
            cur = m.free_energy(np.clip(m.x_of_u(u, P.q), 0, 1), g, P, h, d2, js)
            if cur > prev + 1e-7:
                viol += 1
                maxinc = max(maxinc, cur - prev)
            prev = cur
R['lyapunov'] = {'trials': 50, 'violations': viol, 'max_increase': maxinc}
# Stationarity check via finite differences of F at relaxed points (interior levels only)
res = []
for (g, js, d2) in [(.7, 1, E0), (.5, 1, 3.), (.35, 1, 5.), (1.2, .2, E0)]:
    x = m.relax(np.ones(7) * .5, g, P, EPS, d2, js, T=80, dt=.005)
    for i in range(7):
        if not (1e-3 < x[i] < .98):
            continue
        e = np.zeros(7); e[i] = 1e-6
        res.append(abs(m.free_energy(x + e, g, P, EPS, d2, js) - m.free_energy(x - e, g, P, EPS, d2, js)) / 2e-6)
R['stationarity_max_grad'] = max(res)
R['stationarity_n'] = len(res)


# Monte Carlo: complete-graph Potts heat-bath
def mc_ramp(q, N, bJs, sweeps, reps, seed, ordered):
    rng = np.random.default_rng(seed)
    n = np.zeros((reps, q), int)
    if ordered:
        n[:, 0] = N
    else:
        n[:] = N // q
        n[:, 0] += N - (N // q) * q
    idx, out = np.arange(reps), []
    for bj in bJs:
        for _ in range(sweeps * N):
            pick = (rng.random(reps)[:, None] < np.cumsum(n, 1) / N).argmax(1)
            n[idx, pick] -= 1
            lw = bj * n / N
            lw -= lw.max(1, keepdims=True)
            w = np.exp(lw)
            new = (rng.random(reps)[:, None] < np.cumsum(w, 1) / w.sum(1, keepdims=True)).argmax(1)
            n[idx, new] += 1
        out.append(((q * (n.max(1) / N) - 1) / (q - 1)).mean())
    return np.array(out)


def mf_branch(q, bJs, start):
    x, out = start, []
    for bj in bJs:
        u = float(m.u_of_x(np.array([x]), np.array([float(q)]))[0])
        for _ in range(6000):
            u += .05 * (bj * m.x_of_u(u, q) + 1e-4 - u)
        x = float(m.x_of_u(u, q))
        out.append(x)
    return np.array(out)


bg = np.arange(2.5, 8.51, .25)
A['mc_bJ'] = bg
A['mc_down'] = mc_ramp(7, 600, bg[::-1], 20, 40, 1, True)[::-1]
A['mc_up'] = mc_ramp(7, 600, bg, 20, 40, 2, False)
A['mf_down'] = mf_branch(7, bg[::-1], .99)[::-1]
A['mf_up'] = mf_branch(7, bg, 0.)
R['mc'] = {'N': 600, 'q': 7, 'down': float(bg[np.argmax(A['mc_down'] > .5)]), 'up': float(bg[np.argmax(A['mc_up'] > .5)]), 'mf_lower': m.lower_spinodal(7)[0]}
save('S0')

# ---------- S1 Quasi-static order and hysteresis ----------
gs, Dn, Up, loss, ret = ramp(P)
A['ramp_g'] = gs
R['ramp'] = {'loss': loss, 'ret': ret, 'width': [a - b for a, b in zip(ret, loss)] if None not in ret else None, 'J': P.J, 'q': P.q, 'tau': P.tau,
             'loss_isolated': [m.lower_spinodal(q)[0] / J for q, J in zip(P.q, P.J)], 'ret_isolated': [q / J for q, J in zip(P.q, P.J)]}


def loop_area(P, Tr, gmin=.15, gmax=1.5, dt=.01, d2=E0):
    n = int(Tr / dt)
    gd = np.linspace(gmax, gmin, n)
    x = m.relax(np.ones(7) * .95, gmax, P, EPS, d2)
    u = m.u_of_x(x, P.q)
    xd, xu = np.zeros(n), np.zeros(n)
    for k in range(n):
        x = np.clip(m.x_of_u(u, P.q), 0, 1)
        u = u + dt / P.tau * (m.u_target(x, gd[k], P, EPS, d2) - u)
        xd[k] = x.mean()
    for k in range(n):
        x = np.clip(m.x_of_u(u, P.q), 0, 1)
        u = u + dt / P.tau * (m.u_target(x, gd[::-1][k], P, EPS, d2) - u)
        xu[k] = x.mean()
    return abs(float(np.trapezoid(xd - xu[::-1], -gd)))


la = [{'T': Tr, 'area': loop_area(P, Tr)} for Tr in [5, 10, 20, 50, 100, 200, 500, 1000, 2000]]
R['loop_area'] = la
Ts = np.array([d['T'] for d in la], float)
ar = np.array([d['area'] for d in la])
pop, _ = curve_fit(lambda T, a0, c, p: a0 + c * T ** (-p), Ts[2:], ar[2:], p0=[.3, 1, .6], maxfev=20000)
R['loop_fit'] = {'a_inf': pop[0], 'c': pop[1], 'p': pop[2]}
sc = []
for name, kw in [('q_s=8', dict(q=[8] * 4 + [33, 40, 7])), ('q_s=16', dict(q=[16] * 4 + [33, 40, 7])), ('q_s=20', dict(q=[20] * 4 + [33, 40, 7])), ('q_s=22', dict(q=[22] * 4 + [33, 40, 7])), ('q_s=24', dict(q=[24] * 4 + [33, 40, 7])),
                 ('q_s=60', dict(q=[60] * 4 + [33, 40, 7])), ('eta=0', dict(eta=0.)), ('eta=0.1', dict(eta=.1)), ('eta=0.3', dict(eta=.3)), ('eta=1', dict(eta=1.)),
                 ('kappa=0', dict(kappa=0.)), ('kappa=0.3', dict(kappa=.3)), ('kappa=1', dict(kappa=1.)), ('kappa=2', dict(kappa=2.))]:
    _, _, _, l2, r2 = ramp(m.Params(**kw), dg=.01, T=3., gmax=3.3)
    sc.append({'name': name, 'loss': l2, 'ret': r2, 'loss_mono': bool(None not in l2 and np.all(np.diff(l2) < 0)),
               'ret_mono': bool(None not in r2 and np.all(np.diff(r2) <= 1e-9))})
R['scan'] = sc
R['q_pred_min'] = int(min(q for q in range(3, 61) if m.lower_spinodal(q)[0] / P.J[3] > m.lower_spinodal(33)[0] / P.J[4]))
s = {q: m.lower_spinodal(q)[0] for q in [7, 33, 40]}
R['alphabet'] = {'s': s, 'req': s[40] / s[33], 'J65': float(P.J[5] / P.J[4]),
                 'perm': [{'arr': list(a), 'last_alphabet': int(a[np.argmin([s[q] / j for q, j in zip(a, P.J[4:])])])} for a in itertools.permutations([33, 40, 7])]}
save('S1')

# ---------- S2 Dying path ----------
t, X, G = dying(P, sub=5)
X, G = X[:, 0], G[:, 0]
Rt = np.array([m.evidence_gain(X[k], G[k], P)[0] for k in range(len(t))])
Et = np.array([np.sum(m.entropy(X[k], P.q)) / np.sum(np.log(P.q)) for k in range(len(t))])
A.update(dy_t=t, dy_X=X, dy_G=G, dy_R=Rt, dy_E=Et, dy_a=A0 * np.maximum(1 - t / T0, 0))
A['dy_gflat'] = A['dy_a'] / P.b
dm = rebound_metrics(t, X, G, P)
st = dm['stage_t']
mask = (t > st[5] + 5) & (t < st[6] - 5) & np.isfinite(Rt)
i_ = np.argmin(np.where(mask, Rt, np.inf))
dm['R_entry'] = float(Rt[(t > st[5] + 5) & np.isfinite(Rt)][0])
dm['R_nadir'] = float(Rt[i_])
dm['t_nadir'] = float(t[i_])
after = (t > st[6] + 3) & np.isfinite(Rt)
dm['R_after'] = float(Rt[after].max())
dm['R_wake'] = float(Rt[0])
lr = []
for s0 in st:
    pre = (t > s0 - 12) & (t <= s0 + 1)
    post = (t > s0) & (t < s0 + 15)
    lr.append(float(G[post].max() / G[pre].min()))
dm['local_rebound'] = lr
# Readiness counted over unclipped levels only: the small-signal change of the posterior
Rp = np.array([m.posterior_gain(X[k], G[k], P) for k in range(len(t))])
A['dy_Rp'] = Rp
dm['Rpost_entry'] = float(Rp[(t > st[5] + 5) & np.isfinite(Rp)][0])
dm['Rpost_min'] = float(np.min(Rp[mask & np.isfinite(Rp)]))
dm['Rpost_after'] = float(Rp[after].max())
dm['R_t240'] = float(Rt[np.argmin(abs(t - 240))])
dm['R_t280'] = float(Rt[np.argmin(abs(t - 280))])
dm['t_g_above_peak_until'] = float(t[(t > st[5]) & (t < st[6]) & (G > dm['g_max'])][-1])
dm['t_g_below_trough_from'] = float(t[(t > dm['t_gmax']) & (G < dm['g_min'])][0])
R['dying'] = dm


def R_ratio(t_, X_, G_, Pr, st_, o_, T_):
    """Peak of R after the last collapse relative to its minimum before it; is the flat state clipped?"""
    last_, prev_ = st_[o_[-1]], st_[o_[-2]]
    idx = np.where((t_ > prev_ + .017 * T_) & (t_ < last_ - .017 * T_))[0]
    aft = np.where(t_ > last_ + .01 * T_)[0]
    if len(idx) < 3 or len(aft) < 3:
        return None
    aft = aft[t_[aft] < last_ + .11 * T_]
    idx = idx[::max(1, len(idx) // 300)]
    aft = aft[::max(1, len(aft) // 100)]
    Rn = min(m.evidence_gain(X_[k], G_[k], Pr)[0] for k in idx)
    Ra = max(v for v in (m.evidence_gain(X_[k], G_[k], Pr)[0] for k in aft) if np.isfinite(v))
    return {'R_min': float(Rn), 'R_peak': float(Ra), 'ratio': float(Ra / Rn), 'all_clipped': bool(X_[aft[0]].max() <= 1e-9)}


grid = []
for d_, lam in [(8, 0), (8, 2), (8, 3), (12, 0), (12, 2), (16, 0), (16, 2), (16, 4)]:
    Pl = m.Params(weights=np.exp(lam * np.linspace(0, 1, 7)))
    tt, Xx, Gg = dying(Pl, D=d_, dt=.02, sub=4)
    mm = rebound_metrics(tt, Xx[:, 0], Gg[:, 0], Pl, D=d_)
    rr = R_ratio(tt, Xx[:, 0], Gg[:, 0], Pl, mm['stage_t'], mm['order'], T0)
    grid.append(dict(D=d_, lam=lam, rebound=mm['rebound'], **rr))
R['R_ratio_grid'] = grid
# Assumption A1 relaxed: W taken from the categorical posterior
var = []
for lam in [0, 2]:
    Pl = m.Params(weights=np.exp(lam * np.linspace(0, 1, 7)), categorical_W=True)
    tt, Xx, Gg = dying(Pl, sub=5)
    mm = rebound_metrics(tt, Xx[:, 0], Gg[:, 0], Pl)
    var.append(dict(lam=lam, rebound=mm['rebound'], complete=mm['complete']))
R['A1_variant'] = var
# Weight tilt lambda and mismatch slope D: rebound map
lams = np.array([0, 1, 2, 3, 4])
Ds = np.array([0, 1, 2, 4, 8, 12, 16.])
Rb = np.full((len(lams), len(Ds)), np.nan)
Rc = np.full((len(lams), len(Ds)), np.nan)
for i, lam in enumerate(lams):
    for k, d_ in enumerate(Ds):
        Pl = m.Params(weights=np.exp(lam * np.linspace(0, 1, 7)))
        tt, Xx, Gg = dying(Pl, D=d_, dt=.04)
        mm = rebound_metrics(tt, Xx[:, 0], Gg[:, 0], Pl, D=d_)
        if mm['complete']:
            Rb[i, k], Rc[i, k] = mm['rebound'], mm['rho_c']
A.update(rb_lams=lams, rb_Ds=Ds, rb=Rb, rb_rho=Rc)
Pl = m.Params(weights=np.exp(2 * np.linspace(0, 1, 7)))
tl, Xl, Gl = dying(Pl, sub=5)
Rtl = np.array([m.evidence_gain(Xl[k, 0], Gl[k, 0], Pl)[0] for k in range(len(tl))])
ml = rebound_metrics(tl, Xl[:, 0], Gl[:, 0], Pl)
stl = ml['stage_t']
masl = (tl > stl[5] + 5) & (tl < stl[6] - 5) & np.isfinite(Rtl)
ml['R_nadir'] = float(np.min(Rtl[masl]))
aftl = (tl > stl[6] + 3) & np.isfinite(Rtl)
ml['R_after'] = float(Rtl[aftl].max())
ml['w'] = Pl.w
R['dying_tilted'] = ml
A.update(dyl_t=tl, dyl_G=Gl[:, 0], dyl_R=Rtl)
save('S2')

# ---------- S3 Precision-withdrawal vs prior-relaxation routes ----------
x0w = wake_state(P)
g0w = g_wake(P)


def js_med(t):
    return float(np.interp(t, *zip(*[(0, 1), (40, 1), (160, .2), (200, .2), (220, .28), (280, .28), (340, 1), (420, 1)])))


HEXIT = np.array([.5, .5, .5, .5, 0, 0, 0.])


def hf_med(t):
    return EPS + HW * float(np.interp(t, [0, 20, 40], [1, 1, 0])) + HEXIT * float(np.interp(t, [340, 360], [0, 1]))


tm, Xm, Gm = m.simulate(P, 420, .02, lambda t: A0, lambda t: E0, hf_med, Js_fn=js_med, x0=x0w, g0=g0w, sub=5)
Xm, Gm = Xm[:, 0], Gm[:, 0]
Rm = np.array([m.evidence_gain(Xm[k], Gm[k], P, js_med(tm[k]))[0] for k in range(len(tm))])
Em = np.array([np.sum(m.entropy(Xm[k], P.q)) / np.sum(np.log(P.q)) for k in range(len(tm))])
A.update(med_t=tm, med_X=Xm, med_G=Gm, med_R=Rm, med_E=Em)
back = [float(tm[np.argmax((tm > 205) & (Xm[:, i] > BACK))]) if np.any((tm > 205) & (Xm[:, i] > BACK)) else None for i in range(7)]
R['med'] = {'lost': stage_times(tm, Xm), 'back': back, 'g0': float(Gm[0]), 'g_hold': float(Gm[np.argmin(abs(tm - 190))]),
            'g_min': float(Gm[tm < 200].min()), 'R_hold': float(Rm[np.argmin(abs(tm - 190))]), 'x_end_min': float(Xm[-1].min())}
# rho0 regimes
reg = []
for e in [.2, .5, .83, 1.25, 2., 3.]:
    gw = g_wake(P, A0, e)
    x_ = m.relax(np.ones(7) * .95, gw, P, HW + EPS, e)
    tt, Xx, Gg = m.simulate(P, 240, .02, lambda t: A0, lambda t, e=e: e, lambda t: EPS + HW * float(np.interp(t, [0, 20, 40], [1, 1, 0])),
                            Js_fn=lambda t: float(np.interp(t, [0, 40, 160, 200], [1, 1, .2, .2])), x0=x_, g0=gw)
    reg.append({'e0': e, 'rho0': e * A0, 'g_start': float(Gg[0, 0]), 'g_end': float(Gg[-1, 0]), 'x_end': float(Xx[-1, 0].max())})
R['rho_regimes'] = reg


def route_levelwise(c, T=300, dt=.02):
    n = int(T / dt)
    x, g = x0w.copy(), g0w
    u = m.u_of_x(x, P.q)
    Xr = np.zeros((n, 7))
    for k in range(n):
        tt = k * dt
        js = .2 + .8 / (1 + np.exp((tt - c) / 6))
        h = EPS + HW * float(np.interp(tt, [0, 20, 40], [1, 1, 0]))
        x = np.clip(m.x_of_u(u, P.q), 0, 1)
        u = u + dt / P.tau * (m.u_target(x, g, P, h, E0, js) - u)
        g += dt / P.tau_g * (m.g_star(x, A0, E0, P) - g)
        Xr[k] = x
    return np.arange(n) * dt, Xr


tb, Xb = route_levelwise(np.array([180, 160, 140, 120, 100, 80, 60.]))
R['route_topdown'] = stage_times(tb, Xb)
tb, Xb = route_levelwise(np.array([60, 80, 100, 120, 140, 160, 180.]))
R['route_bottomup'] = stage_times(tb, Xb)

save('S3')

# ---------- S4 Reference states ----------
x_dream = m.relax(np.ones(7) * .95, g0w, P, EPS, E0)
x_sleep = m.relax(np.zeros(7), .2, P, EPS, E0, T=30)
x_m20 = m.relax(np.zeros(7), A0, P, EPS, E0, .2, T=30)
x_m28 = m.relax(np.zeros(7), A0, P, EPS, E0, .28, T=30)


def js_limit(g=A0):
    lo, hi = .05, 1.
    for _ in range(14):
        mid = (lo + hi) / 2
        if m.relax(np.zeros(7), g, P, EPS, E0, mid, T=40).max() > .3:
            hi = mid
        else:
            lo = mid
    return lo


R['js_limit'] = js_limit()
R['gjs_zero_field'] = float(P.q[6] / P.J[6])
k_n = np.argmin(abs(t - dm['t_nadir']))
k_c = np.argmin(abs(t - dm['t_gmax']))
d2_at = lambda tv: E0 + D0 * tv / T0
states = [('wake', 'Waking', x0w, g0w, 1., E0), ('dream', 'Dreaming-like', x_dream, g0w, 1., E0), ('sleep', 'Deep-sleep-like ($g$ = 0.2, assumed)', x_sleep, .2, 1., E0),
          ('nadir', 'Failing support: last stage (only L7 committed)', X[k_n], G[k_n], 1., d2_at(t[k_n])), ('dycl', 'Failing support: after last collapse', X[k_c], G[k_c], 1., d2_at(t[k_c])),
          ('medcl', 'Prior relaxation: hold ($J_s$ = 0.20)', x_m20, A0, .2, E0), ('medcrit', 'Prior relaxation: near instability ($J_s$ = 0.28)', x_m28, A0, .28, E0)]
tab = []
for key, name, xs, g, js, d2 in states:
    Rv, w = m.evidence_gain(xs, g, P, js)
    like7 = P.w[6] * (.5 * g * P.omega * d2 - .5 * np.log(g) + P.c)
    tab.append({'key': key, 'name': name, 'g': g, 'Js': js, 'd2': d2, 'A': xs.mean(), 'x7': xs[6], 'eps': float(np.sum(m.entropy(xs, P.q)) / np.sum(np.log(P.q))), 'R': Rv, 'loop': w, 'net_cost_L7': like7,
                'si': m.si_mib(xs, g, P, js), 'infra': m.band_power(xs, g, P, .05, .5, js), 'delta': m.band_power(xs, g, P, 1, 4, js)})
R['states'] = tab
fs = np.geomspace(.02, 50, 160)
A['spec_f'] = fs
for key, name, xs, g, js, d2 in states:
    Aa, Q = m.ou_system(xs, g, P, js)
    I_ = np.eye(7)
    A['spec_' + key] = np.array([np.real(np.trace(np.linalg.inv(2j * np.pi * f * I_ - Aa) @ Q @ np.linalg.inv(2j * np.pi * f * I_ - Aa).conj().T)) for f in fs])


def barrier_at(js, g=A0, d2=E0):
    xs_ = m.relax(np.zeros(7), g, P, EPS, d2, js, T=30)
    like7 = P.w[6] * (.5 * g * P.omega * d2 - .5 * np.log(g) + P.c)
    return xs_, m.barrier(7, g * P.J[6] * js, g * (P.kappa[5] * xs_[5] + EPS[6]) - like7)[0]


sw = []
for js in [.12, .16, .20, .22, .24, .26, .28]:
    xs_, dF = barrier_at(js)
    Rv, w = m.evidence_gain(xs_, A0, P, js)
    sw.append({'Js': js, 'bJ': A0 * P.J[6] * js, 'x7': xs_[6], 'R': Rv, 'loop': w, 'dF': dF, 'infra': m.band_power(xs_, A0, P, .05, .5, js), 'delta': m.band_power(xs_, A0, P, 1, 4, js), 'si': m.si_mib(xs_, A0, P, js)})
R['tradeoff'] = sw
# size of the term omitted from the precision update, on the metastable branch at each level's collapse
om = []
for i, (qq, jj_, gl) in enumerate(zip(P.q, P.J, loss)):
    xs_, _ = m.lower_spinodal(qq), None
    x_ = m.lower_spinodal(qq)[1]
    U = lambda v: (jj_ / 2) * (1 + (qq - 1) * v ** 2) / qq
    om.append(float(U(x_) - U(0.)))
R['omitted_U_at_spinodal'] = om
R['denominator_range'] = [float(P.b + .5 * P.omega * E0 * P.w.sum()), float(P.b + .5 * P.omega * (E0 + D0) * P.w.sum())]
save('S4')


# ---------- S5 Simulated perturbation: evidence-channel pulse (scaled by g) and direct log-odds pulse (not scaled by g) ----------
def perturb(xs, g, js, d2, site, kind, H=10., dur=.02, kick=1., T_=3., dt=.002, noise=.002, seeds=300, seed=11):
    n, k0 = int(T_ / dt), int(.5 / dt)
    k1 = k0 + int(dur / dt)
    # relax u at the held state first: a clipped level sits at a negative fixed point of u
    u0 = m.u_of_x(xs, P.q)
    for _ in range(int(20 / dt)):
        x = np.clip(m.x_of_u(u0, P.q), 0, 1)
        u0 = u0 + dt / P.tau * (m.u_target(x, g, P, EPS, d2, js) - u0)
    res, resx = [], []
    for arm in [0, 1]:
        rng = np.random.default_rng(seed)
        u = np.tile(u0, (seeds, 1))
        tr = np.zeros((n // 5, seeds, 7))
        trx = np.zeros((n // 5, seeds, 7))
        for k in range(n):
            x = np.clip(m.x_of_u(u, P.q), 0, 1)
            h = EPS.copy()
            if arm and kind == 'evidence' and k0 <= k < k1:
                h[site] += H
            u = u + dt / P.tau * (m.u_target(x, g, P, h, d2, js) - u) + np.sqrt(2 * noise * dt / P.tau) * rng.standard_normal(u.shape)
            if arm and kind == 'direct' and k == k0:
                u[:, site] += kick
            if k % 5 == 0:
                tr[k // 5] = u
                trx[k // 5] = x
        res.append(tr)
        resx.append(trx)
    d = res[1] - res[0]  # common random numbers: trial-wise evoked response
    ev, se = d.mean(1), d.std(1) / np.sqrt(seeds)
    tv = np.arange(n // 5) * dt * 5
    post = tv > .5
    SS = (np.abs(ev[post]) / np.maximum(se[post], 1e-9) > 4) & (np.abs(ev[post]) > .02)
    auc = np.abs(ev[post]).sum(0) * dt * 5
    other = auc.copy()
    other[site] = 0
    aucx = float(np.abs((resx[1] - resx[0]).mean(1)[post]).sum() * dt * 5)
    return {'auc_x': aucx, 'auc': auc, 'auc_total': float(auc.sum()), 'auc_nonsite': float(other.sum()), 'levels': int(SS.any(0).sum()), 'duration': float(SS.any(1).sum() * dt * 5), 'peak_site': float(np.abs(ev[post]).max(0)[site])}, tv, ev


pt = []
for key, name, xs, g, js, d2 in states:
    for site, tag in [(3, 'L4'), (6, 'L7')]:
        for kind in ['evidence', 'direct']:
            r_, tv, ev = perturb(xs, g, js, d2, site, kind)
            r_.update(key=key, site=tag, kind=kind)
            pt.append(r_)
            if tag == 'L7':
                A[f'ev_{kind}_' + key] = ev
A['ev_t'] = tv
R['perturb'] = pt
# Sensitivity of the hold / deep-sleep comparison to the assumed deep-sleep precision
hold_ = [s for s in states if s[0] == 'medcl'][0]
hd, _, _ = perturb(hold_[2], hold_[3], hold_[4], hold_[5], 6, 'direct')
scan = []
for gs in [.1, .2, .24, .26, .28, .3, .32, .34, .35]:
    xs_ = m.relax(np.zeros(7), gs, P, EPS, E0, T=30)
    sd, _, _ = perturb(xs_, gs, 1., E0, 6, 'direct')
    scan.append(dict(g=gs, xmax=float(xs_.max()), R=m.evidence_gain(xs_, gs, P)[0], total_ratio=hd['auc_total'] / sd['auc_total'], nonsite_ratio=hd['auc_nonsite'] / sd['auc_nonsite']))
R['sleep_g_scan'] = scan


def crossing(key):
    for a_, b_ in zip(scan, scan[1:]):
        if a_[key] >= 1 > b_[key]:
            return a_['g'] + (a_[key] - 1) / (a_[key] - b_[key]) * (b_['g'] - a_['g'])
    return None


R['sleep_g_cross'] = {'total': crossing('total_ratio'), 'nonsite': crossing('nonsite_ratio')}
# Dependence of the hold / deep-sleep ratios on pulse size
sl_ = [s for s in states if s[0] == 'sleep'][0]
amp = []
for kind, name_, vals in [('evidence', 'H', [.5, 2, 10, 40]), ('direct', 'kick', [.25, 1, 3])]:
    for v_ in vals:
        a_ = perturb(hold_[2], hold_[3], hold_[4], hold_[5], 6, kind, seeds=100, **{name_: v_})[0]
        b_ = perturb(sl_[2], sl_[3], sl_[4], sl_[5], 6, kind, seeds=100, **{name_: v_})[0]
        amp.append(dict(kind=kind, size=v_, ratio=a_['auc_total'] / b_['auc_total']))
R['pulse_amp'] = amp
# Large direct pulses after the collapse: does L7 re-ignite with g clamped?
rei = []
for key, name, xs, g, js, d2 in states:
    if key not in ('dycl', 'medcl', 'sleep'):
        continue
    for K in [1, 1.5, 2, 2.5, 3, 4, 6]:
        u = m.u_of_x(xs, P.q)
        for _ in range(10000):
            x = np.clip(m.x_of_u(u, P.q), 0, 1)
            u = u + .002 / P.tau * (m.u_target(x, g, P, EPS, d2, js) - u)
        u = u.copy()
        u[6] += K
        for _ in range(15000):
            x = np.clip(m.x_of_u(u, P.q), 0, 1)
            u = u + .002 / P.tau * (m.u_target(x, g, P, EPS, d2, js) - u)
        rei.append(dict(key=key, kick=K, x7=float(np.clip(m.x_of_u(u, P.q), 0, 1)[6])))
R['reignite'] = rei
save('S5')

# ---------- S6 Phase diagram ----------
gg = np.linspace(.05, 1.8, 57)
jj = np.linspace(.05, 1.2, 47)
pa, pb, pr = np.zeros((47, 57)), np.zeros((47, 57)), np.full((47, 57), np.nan)
for i, js in enumerate(jj):
    for k, g in enumerate(gg):
        xa = m.relax(np.ones(7) * .98, g, P, EPS, E0, js, T=3, dt=.01)
        xb = m.relax(np.zeros(7), g, P, EPS, E0, js, T=3, dt=.01)
        pa[i, k], pb[i, k] = xa.mean(), xb.mean()
        if xb.mean() < .05:
            pr[i, k] = m.evidence_gain(xb, g, P, js)[0]
A.update(ph_g=gg, ph_js=jj, ph_a=pa, ph_b=pb, ph_R=pr)
save('S6')

# ---------- S7 Noise: stochastic trajectories and dwell-time statistics ----------
ns = []
for T_n in [.005, .02, .05, .1, .3, 1.]:
    tt, Xx, Gg = dying(P, dt=.02, seeds=200, noise=T_n, seed=5, sub=5)
    taus, full, last7, reb, comp = [], 0, 0, [], 0
    for sd in range(200):
        mm = rebound_metrics(tt, Xx[:, sd], Gg[:, sd], P)
        if not mm['complete']:
            continue
        comp += 1
        stn = mm['stage_t']
        taus.append(kendall_to_canonical(stn))
        full += int(np.all(np.diff(stn) > 0))
        last7 += int(np.argmax(stn) == 6)
        reb.append(mm['rebound'])
    ns.append({'T': T_n, 'complete': comp, 'kendall_median': float(np.median(taus)), 'kendall_min': float(np.min(taus)), 'full_order': full / max(comp, 1), 'last7': last7 / max(comp, 1),
               'rebound_median': float(np.median(reb)), 'rebound_q10': float(np.quantile(reb, .1)), 'rebound_q90': float(np.quantile(reb, .9))})
R['noise_paths'] = ns


def escape(js, T_n=.02, seeds=300, Tmax=4000., dt=.02, seed=8):
    rng = np.random.default_rng(seed)
    u = np.zeros((seeds, 7))
    esc = np.full(seeds, np.nan)
    alive = np.ones(seeds, bool)
    n = int(Tmax / dt)
    for k in range(n):
        x = np.clip(m.x_of_u(u, P.q), 0, 1)
        ut = m.u_target(x, A0, P, EPS, E0, js)
        u = u + dt / P.tau * (ut - u) + np.sqrt(2 * T_n * dt / P.tau) * rng.standard_normal(u.shape)
        hit = alive & (x[:, 6] > .5)
        esc[hit] = k * dt
        alive &= ~hit
        if not alive.any():
            break
    v = esc[~np.isnan(esc)]
    out = {'Js': js, 'T': T_n, 'n': int(v.size), 'censored': int(np.isnan(esc).sum())}
    if v.size > 30:
        t0 = v.min()
        ks = stats.kstest(v - t0, 'expon', args=(0, (v - t0).mean()))
        out.update(mean=float(v.mean()), median=float(np.median(v)), cv=float(v.std() / v.mean()), ks_p=float(ks.pvalue), t_min=float(t0))
    return out, v


dw = []
for js in [.22, .23, .24, .25, .26, .27]:
    o, v = escape(js)
    _, dF = barrier_at(js)
    o['dF'] = dF
    o['bJ'] = A0 * P.J[6] * js
    dw.append(o)
R['dwell'] = dw
save('S7')

# ---------- S8 Lesions ----------
les = []


def lesion_metrics(name, Pq, g_fn=None, note=''):
    tt, Xx, Gg = dying(Pq, dt=.02, g_fn=g_fn)
    mm = rebound_metrics(tt, Xx[:, 0], Gg[:, 0], Pq)
    if Pq.q.max() <= 2:
        gsl, _, _, l_, r_ = ramp(Pq, dg=.01, T=3., gmax=3.3)
    else:
        gsl, _, _, l_, r_ = ramp(Pq, dg=.01, T=3., gmax=3.3)
    full = None not in l_ and None not in r_
    width = [a - b for a, b in zip(r_, l_)] if full else None
    return {'name': name, 'rho_c': mm.get('rho_c'), 'complete': mm['complete'], 'order_ok': bool(mm['complete'] and np.all(np.diff(mm['stage_t']) > 0)), 'rebound': mm.get('rebound'),
            'loss': l_, 'ret': r_, 'loss_mono': bool(None not in l_ and np.all(np.diff(l_) < 0)), 'ret_mono': bool(None not in r_ and np.all(np.diff(r_) <= 1e-9)),
            'first_back': int(np.argmin([v if v is not None else 9e9 for v in r_])), 'width_mean': float(np.mean(width)) if full else None, 'width_max': float(np.max(width)) if full else None}


les.append(lesion_metrics('Full model', m.Params()))
les.append(lesion_metrics('Fixed schedule for g (no precision inference)', m.Params(), g_fn=lambda t: g_wake(P) * max(1 - t / T0, 0.)))
les.append(lesion_metrics('Responsibility weighting removed', m.Params(responsibility=False)))
les.append(lesion_metrics('Binary beliefs (q = 2 at every level)', m.Params(q=[2] * 7)))
les.append(lesion_metrics('No inter-level coupling ($\\kappa$ = 0)', m.Params(kappa=0.)))
les.append(lesion_metrics('No hierarchical gradient ($\\eta$ = 0)', m.Params(eta=0.)))
_t, _X, _ = dying(m.Params(eta=0.), sub=5)
R['eta0_stage_t'] = stage_times(_t, _X[:, 0])
# Without coupling, return needs cheaper commitment, and the net cost changes sign at a computable g
_, _, _, l_k, r_k = ramp(m.Params(kappa=0., c=-2.), dg=.01, T=3., gmax=3.3)
R['kappa0_c2'] = {'loss': l_k, 'ret': r_k}
R['net_cost_zero_g'] = float(brentq(lambda g_: .5 * g_ * E0 - .5 * np.log(g_) + P.c, 1.2, 6))
R['lesions'] = les
save('S8')


# ---------- S9 Broad parameter sweep ----------
def draw(seed):
    r = np.random.default_rng(seed)
    qs, q5, q6, q7 = int(r.integers(4, 61)), int(r.integers(8, 61)), int(r.integers(8, 61)), int(r.integers(3, 61))
    J0, eta, kap, tr = r.uniform(7, 12), r.uniform(0, 1.2), r.uniform(0, 1.5), r.uniform(3, 8)
    lam, c = r.uniform(0, 3), r.uniform(-2, 0)
    a0, e0, D, T, tg = r.uniform(.8, 2), r.uniform(.2, 3), r.uniform(0, 16), r.uniform(150, 600), r.uniform(.5, 8)
    Pr = m.Params(q=[qs] * 4 + [q5, q6, q7], J0=J0, eta=eta, kappa=kap, tau_ratio=tr, c=c, tau_g=tg, weights=np.exp(lam * np.linspace(0, 1, 7)))
    g0 = g_wake(Pr, a0, e0)
    x0_ = m.relax(np.ones(7) * .95, g0, Pr, HW + EPS, e0, T=4)
    res = dict(qs=qs, q5=q5, q6=q6, q7=q7, J0=J0, eta=eta, kappa=kap, lam=lam, c=c, a0=a0, e0=e0, D=D, T=T, tg=tg)
    if x0_.min() < .5:
        res['valid'] = False
        return res
    res['valid'] = True
    tt, Xx, Gg = dying(Pr, T=T, D=D, a0=a0, e0=e0, dt=.03)
    mm = rebound_metrics(tt, Xx[:, 0], Gg[:, 0], Pr, a0=a0, e0=e0, D=D, T=T)
    res['complete'] = mm['complete']
    pred_loss = np.array([m.lower_spinodal(q)[0] / J for q, J in zip(Pr.q, Pr.J)])
    pred_ret = Pr.q / Pr.J
    res['pred_last'] = int(np.argmin(pred_loss))
    res['pred_first_back'] = int(np.argmin(pred_ret))
    res['pred_mono'] = bool(np.all(np.diff(pred_loss) < 0))
    if mm['complete']:
        res.update(rebound=mm['rebound'], rho_c=mm['rho_c'], order=mm['order'], last=mm['order'][-1], canonical=bool(np.all(np.diff(mm['stage_t']) > 0)))
        res['tau_pred'] = float(stats.kendalltau(-pred_loss, mm['stage_t']).statistic)
        rr = R_ratio(tt, Xx[:, 0], Gg[:, 0], Pr, mm['stage_t'], mm['order'], T)
        if rr:
            res.update(R_ratio=rr['ratio'], all_clipped=rr['all_clipped'])
    _, _, _, l_, r_ = ramp(Pr, d2=e0, dg=.02, T=1.6, gmax=max(2., 2 * g0), dt=.01)
    if None not in l_ and None not in r_:
        rmin = min(r_)
        res.update(ramp_ok=True, first_back_set=[i for i, v in enumerate(r_) if v <= rmin + 1e-9], ret_mono=bool(np.all(np.diff(r_) <= 1e-9)),
                   ret_strict=bool(np.all(np.diff(r_) < -1e-9)), ret_below_wake=bool(max(r_) <= g0),
                   loss_mono=bool(np.all(np.diff(l_) < 1e-9)), hyst=bool(all(a > b for a, b in zip(r_, l_))))
    else:
        gq = float(pred_ret.min())
        res.update(ramp_ok=False, loss_fail=None in l_, ret_fail=None in r_, cost_at_minqJ=float(.5 * gq * e0 - .5 * np.log(gq) + c))
    return res


with Pool(2) as pool:
    dr = pool.map(draw, range(800))
R['sweep_raw_n'] = len(dr)
v = [d for d in dr if d['valid']]
c_ = [d for d in v if d.get('complete')]
rm = [d for d in v if d.get('ramp_ok')]
reb = np.array([d['rebound'] for d in c_])
rho = np.array([d['rho_c'] for d in c_])
both = [d for d in rm if d.get('complete')]
R['sweep'] = {'n': len(dr), 'valid': len(v), 'complete': len(c_), 'ramp': len(rm),
              'last_equals_pred': float(np.mean([d['last'] == d['pred_last'] for d in c_])),
              'tau_pred_median': float(np.median([d['tau_pred'] for d in c_])),
              'tau_pred_q10': float(np.quantile([d['tau_pred'] for d in c_], .1)),
              'canonical_given_pred': float(np.mean([d['canonical'] for d in c_ if d['pred_mono']])),
              'n_pred_mono': int(sum(d['pred_mono'] for d in c_)),
              'canonical_overall': float(np.mean([d['canonical'] for d in c_])),
              'first_back_has_pred': float(np.mean([d['pred_first_back'] in d['first_back_set'] for d in rm])),
              'first_back_has_last': float(np.mean([d['last'] in d['first_back_set'] for d in both])),
              'hyst': float(np.mean([d['hyst'] for d in rm])),
              'ret_mono': float(np.mean([d['ret_mono'] for d in rm])),
              'ret_mono_given_canonical': float(np.mean([d['ret_mono'] for d in both if d['canonical']])),
              'n_both_canonical': int(sum(d['canonical'] for d in both)),
              'first_back_L7_given_canonical': float(np.mean([6 in d['first_back_set'] for d in both if d['canonical']])),
              'no_return': int(len(v) - len(rm)),
              'fail_ret': int(sum(d['ret_fail'] for d in v if not d.get('ramp_ok'))),
              'fail_cost_pos': int(sum(d['cost_at_minqJ'] > 0 for d in v if not d.get('ramp_ok'))),
              'ret_strict': float(np.mean([d['ret_strict'] for d in rm])),
              'ret_below_wake_of_valid': float(sum(d['ret_below_wake'] for d in rm) / len(v)),
              'n_R_ratio': int(sum('R_ratio' in d for d in c_)),
              'R_ratio_gt1': float(np.mean([d['R_ratio'] > 1 for d in c_ if 'R_ratio' in d])),
              'R_ratio_gt1p1': float(np.mean([d['R_ratio'] > 1.1 for d in c_ if 'R_ratio' in d])),
              'R_ratio_all_clipped': float(np.mean([d['all_clipped'] for d in c_ if 'R_ratio' in d])),
              'R_ratio_spearman_reb': float(stats.spearmanr([d['R_ratio'] for d in c_ if 'R_ratio' in d], [d['rebound'] for d in c_ if 'R_ratio' in d]).statistic),
              'rebound_gt1_given_rho_gt1': float(np.mean(reb[rho > 1] > 1.005)), 'rebound_gt1_given_rho_lt1': float(np.mean(reb[rho < 1] > 1.005)),
              'n_rho_gt1': int((rho > 1).sum()), 'n_rho_lt1': int((rho < 1).sum()), 'rebound_median_rho_gt1': float(np.median(reb[rho > 1])),
              'rebound_q90_rho_gt1': float(np.quantile(reb[rho > 1], .9)), 'rebound_gt1_given_rho_gt2': float(np.mean(reb[rho > 2] > 1.005)), 'n_rho_gt2': int((rho > 2).sum()),
              'spearman_reb_rho': float(stats.spearmanr(rho, reb).statistic)}
A['sw_rho'] = rho
A['sw_reb'] = reb
save('S9 done')
print('ALL DONE', flush=True)
