# -*- coding: utf-8 -*-
"""Hierarchical Potts active-inference model with inferred precision.

Each level i holds a categorical belief over q_i states, written as a mixture
q_i = x_i * delta_win + (1 - x_i) * Uniform, so x_i is the responsibility of the
committed component.

Variational free energy at fixed precision g:
  F(x|g) = sum_i [ -(g J_i Js / 2) (1+(q_i-1)x_i^2)/q_i - S(x_i;q_i) ]
           - g kappa sum_i x_i x_{i+1} - g sum_i h_i x_i
           + sum_i w_i x_i [ (1/2) g omega d^2 - (1/2) ln g + c ]
Belief dynamics (natural-gradient descent in the log-odds u=ln(p/r)):
  tau_i du_i/dt = u*_i(x,g) - u_i
Precision: Gamma(a,b) hyperprior, responsibility-weighted Gaussian likelihood:
  g* = (a + W/2) / (b + omega d^2 W / 2),  W = sum_i w_i x_i
  tau_g dg/dt = g* - g
"""
import numpy as np
EPS = 1e-12
LEVELS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']


class Params:
    def __init__(self, q=None, J0=9.0, eta=0.68, kappa=0.6, tau_ratio=5.0, tau0=0.06, omega=1.0, b=1.0, c=-1.0,
                 weights=None, tau_g=2.0, N=7, responsibility=True, categorical_W=False):
        self.N = N
        self.q = np.array(q if q is not None else [33, 33, 33, 33, 33, 40, 7], float)
        self.h = np.linspace(0, 1, N)
        self.J = J0 * (1 + eta * self.h)
        self.kappa = np.full(N - 1, kappa) if np.isscalar(kappa) else np.array(kappa, float)
        self.tau = tau0 * tau_ratio ** self.h
        self.omega, self.b, self.c, self.tau_g = omega, b, c, tau_g
        self.w = np.ones(N) / N if weights is None else np.array(weights, float) / np.sum(weights)
        self.responsibility = responsibility
        self.categorical_W = categorical_W  # relaxes A1: W from the categorical posterior, not the indicator


def x_of_u(u, q):
    e = np.exp(np.clip(u, -40, 40))
    return (e - 1) / (e + q - 1)


def u_of_x(x, q):
    p = np.clip((1 + (q - 1) * x) / q, EPS, 1)
    r = np.clip((1 - x) / q, EPS, 1)
    return np.log(p / r)


def dxdu(x, q):
    e = np.exp(np.clip(u_of_x(x, q), -40, 40))
    return e * q / (e + q - 1) ** 2


def entropy(x, q):
    p = np.clip((1 + (q - 1) * x) / q, EPS, 1)
    r = np.clip((1 - x) / q, EPS, 1)
    return -(p * np.log(p) + (q - 1) * r * np.log(r))


def neigh(x, kappa):
    nb = np.zeros_like(x)
    nb[..., 1:] += kappa * x[..., :-1]
    nb[..., :-1] += kappa * x[..., 1:]
    return nb


def u_target(x, g, P, h, d2, Js=1.0):
    """Right-hand side; identical fixed point to dF/dx = 0. Js may be scalar or per level."""
    q = P.q
    lg = np.log(np.maximum(g, 1e-6))
    like = P.w * (0.5 * g * P.omega * d2 - 0.5 * lg + P.c)
    return g * P.J * Js * x + q / (q - 1) * (g * (neigh(x, P.kappa) + h) - like)


def g_star(x, a, d2, P):
    W = np.sum(P.w * x, axis=-1)
    if P.categorical_W:
        W = np.sum(P.w * (1 + (P.q - 1) * x) / P.q, axis=-1)
    if P.responsibility:
        return (a + 0.5 * W) / (P.b + 0.5 * P.omega * d2 * W)
    return (a + 0.5) / (P.b + 0.5 * P.omega * d2)  # lesion: every level pays regardless of commitment


def free_energy(x, g, P, h, d2, Js=1.0):
    q = P.q
    lg = np.log(np.maximum(g, 1e-6))
    F = np.sum(-(g * P.J * Js / 2) * (1 + (q - 1) * x ** 2) / q - entropy(x, q))
    F -= g * np.sum(P.kappa * x[:-1] * x[1:]) + g * np.sum(h * x)
    F += np.sum(P.w * x * (0.5 * g * P.omega * d2 - 0.5 * lg + P.c))
    return float(F)


def relax(x0, g, P, h, d2, Js=1.0, T=4.0, dt=0.01):
    u = u_of_x(x0, P.q)
    for _ in range(int(T / dt)):
        x = np.clip(x_of_u(u, P.q), 0, 1)  # x >= 0: the winner is by definition the most probable category
        u = u + dt / P.tau * (u_target(x, g, P, h, d2, Js) - u)
    return np.clip(x_of_u(u, P.q), 0, 1)


def simulate(P, T, dt, a_fn, d2_fn, h_fn, Js_fn=None, x0=None, g0=None, noise=0.0, seeds=1, seed=0, g_fn=None,
             sub=1):
    """Integrate belief and precision dynamics; vectorised over `seeds` trajectories."""
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    x = np.tile(np.ones(P.N) * 0.95 if x0 is None else x0, (seeds, 1))
    u = u_of_x(x, P.q)
    g = np.full(seeds, g_star(x[0], a_fn(0), d2_fn(0), P) if g0 is None else g0)
    m = n // sub
    X = np.zeros((m, seeds, P.N))
    G = np.zeros((m, seeds))
    ts = np.zeros(m)
    for k in range(n):
        t = k * dt
        js = 1.0 if Js_fn is None else Js_fn(t)
        h = h_fn(t)
        d2 = d2_fn(t)
        x = np.clip(x_of_u(u, P.q), 0, 1)
        lg = np.log(np.maximum(g, 1e-6))[:, None]
        gg = g[:, None]
        like = P.w * (0.5 * gg * P.omega * d2 - 0.5 * lg + P.c)
        ut = gg * P.J * js * x + P.q / (P.q - 1) * (gg * (neigh(x, P.kappa) + h) - like)
        du = dt / P.tau * (ut - u)
        if noise > 0:
            du = du + np.sqrt(2 * noise * dt / P.tau) * rng.standard_normal(x.shape)
        u = u + du
        if g_fn is not None:
            g = np.full(seeds, g_fn(t))
        else:
            g = g + dt / P.tau_g * (g_star(x, a_fn(t), d2, P) - g)
        g = np.maximum(g, 1e-4)
        if k % sub == 0 and k // sub < m:
            X[k // sub], G[k // sub], ts[k // sub] = x, g, t
    return ts, X, G


def hessian(x, g, P, Js=1.0):
    q = P.q
    p = np.clip((1 + (q - 1) * x) / q, EPS, 1)
    r = np.clip((1 - x) / q, EPS, 1)
    d = (q - 1) / q * ((q - 1) / (q * p) + 1 / (q * r)) - g * P.J * Js * (q - 1) / q
    H = np.diag(d)
    for i in range(P.N - 1):
        H[i, i + 1] = H[i + 1, i] = -g * P.kappa[i]
    return H


def loop_matrix(x, g, P, Js=1.0):
    q = P.q
    d = dxdu(x, q)
    d = np.where(x <= 1e-9, 0.0, d)  # a level clipped at x = 0 has no local response: dx/du = 0
    D = np.diag(d)
    Adj = np.zeros((P.N, P.N))
    for i in range(P.N - 1):
        Adj[i, i + 1] = Adj[i + 1, i] = P.kappa[i]
    G = g * (np.diag(P.J * Js * np.ones(P.N)) + np.diag(q / (q - 1)) @ Adj)
    return G @ D


def evidence_gain(x, g, P, Js=1.0):
    """R: change in log-odds u per unit of sensory evidence (nat), averaged over levels.
    du = (I - M)^{-1} g q/(q-1) dh; returns (R, largest eigenvalue of M).
    At a level clipped at x = 0 this is a sub-threshold drive: see posterior_gain."""
    M = loop_matrix(x, g, P, Js)
    w = np.linalg.eigvals(M).real.max()
    if w >= 1:
        return np.inf, float(w)
    return float(np.linalg.solve(np.eye(P.N) - M, g * P.q / (P.q - 1)).mean()), float(w)


def lower_spinodal(q):
    xs = np.linspace(1e-4, 1 - 1e-6, 400001)
    bj = np.log((1 + (q - 1) * xs) / (1 - xs)) / xs
    i = np.argmin(bj)
    return float(bj[i]), float(xs[i])


def coexistence(q):
    return 2 * (q - 1) / (q - 2) * np.log(q - 1)


def F_single(x, q, bJ):
    return -(bJ / 2) * (1 + (q - 1) * x ** 2) / q - entropy(x, q)


def barrier(q, bJ, field=0.0):
    """Free-energy barrier from the contentless minimum to the saddle, with net evidence `field` conjugate to x."""
    xs = np.linspace(0, 1 - 1e-6, 400001)
    f = F_single(xs, q, bJ) - field * xs
    d = np.diff(f)
    i0 = int(np.argmax(d > 0)) if np.any(d > 0) else None   # first minimum
    if i0 is None:
        return 0.0, 0.0, 0.0
    rest = d[i0:]
    if not np.any(rest < 0):
        return 0.0, float(xs[i0]), float(xs[i0])
    i1 = i0 + int(np.argmax(rest < 0))                      # next maximum
    return float(f[i1] - f[i0]), float(xs[i0]), float(xs[i1])


def posterior_gain(x, g, P, Js=1.0):
    """As evidence_gain, counting only unclipped levels: the small-signal change of the posterior itself."""
    M = loop_matrix(x, g, P, Js)
    if np.linalg.eigvals(M).real.max() >= 1:
        return np.inf
    du = np.linalg.solve(np.eye(P.N) - M, g * P.q / (P.q - 1))
    return float(np.where(x > 1e-9, du, 0.).mean())


def ou_system(x, g, P, Js=1.0, T=1.0):
    """Linearisation of the simulated log-odds dynamics: d(du) = tau^{-1}(M - I) du dt + noise, noise variance 2T dt/tau."""
    M = loop_matrix(x, g, P, Js)
    Ti = np.diag(1 / P.tau)
    return Ti @ (M - np.eye(P.N)), 2 * T * Ti


def si_mib(x, g, P, Js=1.0, lag=0.05):
    """Stochastic interaction of the linearised log-odds process at the minimum-information bipartition (Ay 2015). A proxy, not IIT 4.0 Phi."""
    from scipy.linalg import expm, solve_continuous_lyapunov
    from itertools import combinations
    A, Q = ou_system(x, g, P, Js)
    if np.linalg.eigvals(A).real.max() >= 0:
        return np.nan
    S = solve_continuous_lyapunov(A, -Q)
    C = expm(A * lag) @ S
    n = P.N

    def cl(idx):
        idx = list(idx)
        Sii, Cii = S[np.ix_(idx, idx)], C[np.ix_(idx, idx)]
        return np.linalg.slogdet(Sii - Cii @ np.linalg.solve(Sii, Cii.T))[1]

    Hw = cl(range(n))
    best, val = np.inf, None
    for k in range(1, n // 2 + 1):
        for a in combinations(range(n), k):
            b = [i for i in range(n) if i not in a]
            si = 0.5 * (cl(a) + cl(b) - Hw)
            ma = 0.5 * (np.linalg.slogdet(S[np.ix_(a, a)])[1] - cl(a))
            mb = 0.5 * (np.linalg.slogdet(S[np.ix_(b, b)])[1] - cl(b))
            nv = si / max(min(ma, mb), 1e-12)
            if nv < best:
                best, val = nv, si
    return float(val)


def band_power(x, g, P, f1, f2, Js=1.0, nf=300):
    A, Q = ou_system(x, g, P, Js)
    fs = np.geomspace(f1, f2, nf)
    I = np.eye(P.N)
    S = [np.real(np.trace(np.linalg.inv(2j * np.pi * f * I - A) @ Q @ np.linalg.inv(2j * np.pi * f * I - A).conj().T)) for f in fs]
    return float(np.trapezoid(S, fs))
