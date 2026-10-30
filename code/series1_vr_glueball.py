#!/usr/bin/env python3
"""Series 1 of the data note: SU(2) Wilson action, beta 2.3 (8^4, 12^4, 16^4) and 2.4 (12^4, 16^4), T = L.
0++ glueball and static potential on the original (ORIG), center-projected (CP) and vortex-removed (VR) links in direct
maximal center gauge (one gauge copy, over-relaxation, numpy). This is the script that produced data/series1_summary.csv.
It is an earlier version of vr_glueball.py (same algorithms; vr_glueball.py adds T != L and a numba gauge fixing).

Modes:
  selftest
  run L beta seed deadline_unix   -> runs_series1/run_L{L}_b{beta}_s{seed}.npz (+ .ckpt)
  analyze                         -> runs_series1/analyze.log, runs_series1/summary.csv, PNGs in runs_series1/
  long24                          -> runs_series1/long24.log, runs_series1/long24.csv: E1, E1', E2 and Creutz ratios on the
                                     stored beta 2.4 Wilson loops (analysis added after referee review, not pre-registered)
Inputs: production streams in runs_series1/, earlier exploratory streams (beta 2.3 8^4 and 12^4, beta 2.4 12^4, same
action and gauge fixing, written by an earlier script with 3 smearing levels) in runs_series1/exploratory/. The
exploratory measurement files are not shipped; the analysis reads only their stored operators, Wilson loops and plaquettes.
"""
import sys, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_series1")
QOUT = os.path.join(OUT, "exploratory")   # earlier exploratory streams, read only
os.makedirs(OUT, exist_ok=True)

A_SQS_LIT = {2.3: 0.369, 2.4: 0.266, 2.5: 0.188}   # only to scale smearing steps
ALPHA_APE = 0.5
SMEAR_K = (0, 2, 4, 8, 12)
NOR = 4
NSKIP = 2
ENS = ("ORIG", "VR", "CP")


def sh(A, d, s):
    return np.roll(A, -s, axis=A.ndim - 4 + d)


def mul(a, b):
    a0, av = a[0], a[1:]
    b0, bv = b[0], b[1:]
    c0 = a0 * b0 - (av * bv).sum(0)
    cv = a0 * bv + b0 * av - np.stack([av[1] * bv[2] - av[2] * bv[1],
                                       av[2] * bv[0] - av[0] * bv[2],
                                       av[0] * bv[1] - av[1] * bv[0]])
    return np.concatenate([c0[None], cv])


def dag(a):
    return np.concatenate([a[:1], -a[1:]])


def norm(a):
    return np.sqrt((a * a).sum(0))


def staple_paths(U, mu, dirs):
    S = 0
    for nu in dirs:
        if nu == mu:
            continue
        S = S + mul(mul(U[nu], sh(U[mu], nu, 1)), dag(sh(U[nu], mu, 1)))
        Ud = sh(U[nu], nu, -1)
        S = S + mul(mul(dag(Ud), sh(U[mu], nu, -1)), sh(Ud, mu, 1))
    return S


class Lattice:
    def __init__(self, L, seed=0):
        self.L = L
        self.rng = np.random.default_rng(seed)
        U = self.rng.normal(size=(4, 4) + (L,) * 4)
        self.U = U / np.sqrt((U * U).sum(1, keepdims=True))
        g = np.indices((L,) * 4).sum(0) % 2
        self.par = [g == 0, g == 1]

    def _kp(self, alpha):
        x0 = np.empty_like(alpha); todo = np.arange(alpha.size)
        while todo.size:
            a = alpha[todo]
            r1, r2, r3, r4 = (1 - self.rng.random((4, todo.size)))
            lam2 = -(np.log(r1) + np.cos(2 * np.pi * r2) ** 2 * np.log(r3)) / (2 * a)
            ok = r4 ** 2 <= 1 - lam2
            x0[todo[ok]] = 1 - 2 * lam2[ok]
            todo = todo[~ok]
        return x0

    def sweep(self, beta, nor=NOR):
        for mu in range(4):
            for p in self.par:
                V = staple_paths(self.U, mu, range(4))[:, p]
                k = norm(V); Vt = V / k
                x0 = self._kp(beta * k)
                v = self.rng.normal(size=(3, x0.size)); v /= norm(v)
                X = np.concatenate([x0[None], np.sqrt(np.clip(1 - x0 ** 2, 0, None)) * v])
                self.U[mu][:, p] = mul(X, Vt)
        for _ in range(nor):
            for mu in range(4):
                for p in self.par:
                    V = staple_paths(self.U, mu, range(4))[:, p]
                    Vt = V / norm(V)
                    self.U[mu][:, p] = mul(mul(Vt, dag(self.U[mu][:, p])), Vt)
        self.U[:] /= np.sqrt((self.U * self.U).sum(1, keepdims=True))


def ape(Us, nstep):
    U = Us.copy()
    full = lambda U: np.concatenate([np.zeros_like(U[:1]), U])
    for _ in range(nstep):
        F = full(U)
        new = [(1 - ALPHA_APE) * F[i] + ALPHA_APE / 4 * staple_paths(F, i, (1, 2, 3)) for i in (1, 2, 3)]
        U = np.stack(new); U /= norm(U.swapaxes(0, 1))[:, None]
    return U


def line(F, d, n):
    P = F[d]
    for s in range(1, n):
        P = mul(P, sh(F[d], d, s))
    return P


def shapes(L):
    return ([(r, r) for r in range(1, L // 2 + 1)] + [(r, r + 1) for r in range(1, L // 2)]
            + [s for s in ((1, 3), (2, 4), (3, 6)) if max(s) <= L // 2])


def loop_ops(F, L):
    out = []
    lines = {(d, n): line(F, d, n) for d in (1, 2, 3) for n in range(1, L // 2 + 2)}
    for (r1, r2) in shapes(L):
        acc = np.zeros(L)
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if i == j or (r1 == r2 and j < i):
                    continue
                W = mul(mul(lines[i, r1], sh(lines[j, r2], i, r1)), dag(mul(lines[j, r2], sh(lines[i, r1], j, r2))))
                acc += W[0].sum((1, 2, 3))
        out.append(acc)
    return np.array(out)


def wilson(F, U0, L, Tmax):
    W = np.zeros((L // 2, Tmax))
    tl = {T: line(np.concatenate([U0[None], F[1:]]), 0, T) for T in range(1, Tmax + 1)}
    for R in range(1, L // 2 + 1):
        for i in (1, 2, 3):
            S = line(F, i, R)
            for T in range(1, Tmax + 1):
                w = mul(mul(S, sh(tl[T], i, R)), dag(mul(tl[T], sh(S, 0, T))))
                W[R - 1, T - 1] += w[0].mean() / 3
    return W


def nsmear(beta):
    f = (A_SQS_LIT[2.3] / A_SQS_LIT[beta]) ** 2
    return [int(round(k * f)) for k in SMEAR_K]


def plaq_field(U):
    """(6, L^4) tr/2 of every plaquette."""
    return np.array([mul(mul(U[m], sh(U[n], m, 1)), mul(dag(sh(U[m], n, 1)), dag(U[n])))[0]
                     for m in range(4) for n in range(m + 1, 4)])


def mcg_functional(U):
    return (U[:, 0] ** 2).mean()


def mcg(U, par, omega=1.85, tol=1e-8, maxit=3000):
    """direct maximal center gauge. Site x: F_x(g) = g^T M g, M = sum_mu d d^T + u u^T, d = dag U_mu(x), u = U_mu(x-mu)
    (Re tr(g U)/2 = <g, dag U>, Re tr(U g^dag)/2 = <g, U>). Exact local max = top eigenvector; over-relax g -> g^omega."""
    U = U.copy(); F0 = mcg_functional(U); Fprev = F0
    for it in range(maxit):
        for p in par:
            M = 0
            for mu in range(4):
                d = dag(U[mu])[:, p]; u = sh(U[mu], mu, -1)[:, p]
                M = M + np.einsum("in,jn->nij", d, d) + np.einsum("in,jn->nij", u, u)
            g = np.linalg.eigh(M)[1][:, :, -1].T                      # (4, n)
            g *= np.where(g[0] < 0, -1, 1)
            if omega != 1:
                th = np.arccos(np.clip(g[0], -1, 1)); nv = g[1:] / np.maximum(np.sin(th), 1e-300)
                g = np.concatenate([np.cos(omega * th)[None], np.sin(omega * th) * nv])
                g /= norm(g)
            G = np.zeros_like(U[0]); G[0] = 1; G[:, p] = g
            U = np.stack([mul(mul(G, U[mu]), dag(sh(G, mu, 1))) for mu in range(4)])
        U /= norm(U.swapaxes(0, 1))[:, None]
        Fn = mcg_functional(U)
        if abs(Fn - Fprev) < tol * Fn:
            break
        Fprev = Fn
    return U, F0, Fn, it + 1


def center_split(U, par):
    Ug, F0, F1, nit = mcg(U, par)
    Z = np.sign(Ug[:, 0])                                   # (4, L^4) center element per link
    VR = Ug * Z[:, None]
    CP = np.zeros_like(Ug); CP[:, 0] = Z
    return {"ORIG": U, "VR": VR, "CP": CP}, (F0, F1, nit)


def measure(lat, beta):
    L = lat.L; ns = nsmear(beta)
    ens, gf = center_split(lat.U, lat.par)
    res = {}
    for name, U in ens.items():
        P = plaq_field(U)
        levels = [0] if name == "CP" else ns
        ops, Us, done, Wl = [], U[1:].copy(), 0, None
        for n in levels:
            Us = ape(Us, n - done); done = n
            F = np.concatenate([U[:1], Us])
            ops.append(loop_ops(F, L))
            if n == levels[min(1, len(levels) - 1)]:
                Wl = wilson(F, U[0], L, L // 2)
        res[name] = (np.array(ops), Wl, P.mean(), (P < 0).mean() if name == "CP" else np.nan)
    return res, gf


def run(L, beta, seed, deadline, ntherm=300, ckevery=10):
    import pickle, time
    fn = f"{OUT}/run_L{L}_b{beta}_s{seed}.npz"; ck = fn[:-4] + ".ckpt"
    if os.path.exists(ck):
        lat, D, vd, gfs, ts = pickle.load(open(ck, "rb")); print(f"resumed at {len(vd)} meas", flush=True)
    else:
        lat = Lattice(L, seed=seed)
        for _ in range(ntherm): lat.sweep(beta)
        D = {f"{e}_{k}": [] for e in ENS for k in ("O", "W", "P")}; vd, gfs, ts = [], [], []
    while time.time() < deadline:
        for _ in range(NSKIP): lat.sweep(beta)
        assert np.allclose(norm(lat.U.swapaxes(0, 1)), 1, atol=1e-10), "links left SU(2)"
        res, gf = measure(lat, beta)
        assert gf[2] >= gf[1], "gauge fixing decreased the functional"
        for e in ENS:
            D[f"{e}_O"].append(res[e][0].astype(np.float32)); D[f"{e}_W"].append(res[e][1]); D[f"{e}_P"].append(res[e][2])
        vd.append(res["CP"][3]); gfs.append(gf); ts.append(time.time())
        i = len(vd)
        if i % ckevery == 0:
            pickle.dump((lat, D, vd, gfs, ts), open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
            np.savez(fn + ".tmp.npz", **{k: np.array(v) for k, v in D.items()}, vdens=np.array(vd), gf=np.array(gfs),
                     ts=np.array(ts), L=L, beta=beta, nsmear=nsmear(beta), shapes=np.array(shapes(L)))
            os.replace(fn + ".tmp.npz", fn)
            print(f"L={L} b={beta} s={seed} {i} P={np.mean(D['ORIG_P']):.5f} PVR={np.mean(D['VR_P']):.5f} "
                  f"PCP={np.mean(D['CP_P']):.5f} vdens={np.mean(vd):.4f} gf_it={gf[2]} dt={ts[-1]-ts[-2] if i>1 else 0:.1f}s",
                  flush=True)
    print("deadline reached", flush=True)


def selftest():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 5)); a /= norm(a); b = rng.normal(size=(4, 5)); b /= norm(b)
    M = lambda q: np.array([[q[0] + 1j * q[3], q[2] + 1j * q[1]], [-q[2] + 1j * q[1], q[0] - 1j * q[3]]])
    for n in range(5):
        assert np.allclose(M(mul(a, b)[:, n]), M(a[:, n]) @ M(b[:, n]))
    # local MCG identity: Re tr(gU)/2 = <g, dag U>, Re tr(U g^dag)/2 = <g, U>
    assert np.allclose(mul(a, b)[0], (a * dag(b)).sum(0)) and np.allclose(mul(b, dag(a))[0], (a * b).sum(0))
    lat = Lattice(6, seed=1)
    for _ in range(30): lat.sweep(2.3)
    assert np.allclose(norm(lat.U.swapaxes(0, 1)), 1, atol=1e-12), "links left SU(2)"
    # pure (omega=1) MCG steps never decrease the functional; OR version converges higher than start
    U1, F0, F1, _ = mcg(lat.U, lat.par, omega=1, maxit=1)
    U2, _, F2, _ = mcg(U1, lat.par, omega=1, maxit=1)
    assert F0 < F1 <= F2 + 1e-14, (F0, F1, F2)
    ens, (F0, Fend, nit) = center_split(lat.U, lat.par)
    print(f"MCG: functional {F0:.4f} -> {Fend:.4f} in {nit} it (omega=1 one step {F1:.4f})")
    assert Fend > F1 and Fend > 0.5
    Ug = ens["VR"] * np.sign(ens["CP"][:, :1])
    assert np.allclose(norm(Ug.swapaxes(0, 1)), 1, atol=1e-12)
    # gauge fixing leaves the gauge-invariant plaquette field untouched
    assert np.allclose(plaq_field(Ug), plaq_field(lat.U), atol=1e-10), "MCG is not a gauge transformation"
    # center factorization per plaquette: P_orig = Z_P * P_VR
    assert np.allclose(plaq_field(lat.U), plaq_field(ens["CP"]) * plaq_field(ens["VR"]), atol=1e-10)
    assert set(np.unique(plaq_field(ens["CP"]))) <= {-1.0, 1.0}
    # loop operators gauge invariant
    F = np.concatenate([lat.U[:1], ape(lat.U[1:], 2)]); Fg = np.concatenate([Ug[:1], ape(Ug[1:], 2)])
    assert np.allclose(loop_ops(F, 6), loop_ops(Fg, 6))
    # plaquette at beta=2.3 vs literature 0.6024
    p = np.mean([(lat.sweep(2.3), plaq_field(lat.U).mean())[1] for _ in range(60)])
    print(f"<P>(beta=2.3, 6^4) = {p:.4f} (lit 0.6024)")
    assert abs(p - 0.6024) < 0.004, p
    print("selftest OK")



# ---------------- analysis (estimators fixed before the production data) ----------------
QLEV = {2.3: (0, 2, 8), 2.4: (0, 4, 15)}          # exploratory-stream smearing levels = basis A levels
ASHAPES = [(1, 1), (2, 2), (3, 3), (4, 4), (1, 2), (2, 3), (3, 4)]
COL = {"ORIG": "#2a78d6", "VR": "#eb6834", "CP": "#1baf7a"}
MRK = {"ORIG": "o", "VR": "s", "CP": "^"}
LIT_ASQS = {2.3: 0.369, 2.4: 0.266}
LIT_PLAQ = {2.3: 0.6024, 2.4: 0.6300}
PRUNE = 0                                  # deviation: GEVP t0 = 1 dropped (paper sec. 1), pruning unused


def tau_int(x, c=6.0):
    """Madras-Sokal windowed integrated autocorrelation time (in measurements)."""
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:n]; acf /= acf[0]
    tau = 0.5
    for W in range(1, n // 2):
        tau += acf[W]
        if W >= c * tau:
            break
    return max(tau, 0.5)


def basis_index(d, beta, basis, e):
    """(level indices, shape indices) of the stored operator array for basis A or B."""
    shp = [tuple(x) for x in d["shapes"]]; lev = list(d["nsmear"])
    if e == "CP":
        li = [0]
    else:
        li = [lev.index(k) for k in QLEV[beta]] if basis == "A" else list(range(len(lev)))
    si = [shp.index(s) for s in ASHAPES] if basis == "A" else list(range(len(shp)))
    return li, si


def load_groups():
    groups = {}
    for src, pat in (("Q", f"{QOUT}/run_*.npz"), ("Q2", f"{OUT}/run_*.npz")):
        for fn in sorted(glob.glob(pat)):
            if fn.endswith(".tmp.npz"):
                continue
            d = dict(np.load(fn)); d["src"] = src; d["name"] = os.path.basename(fn)
            groups.setdefault((float(d["beta"]), int(d["L"])), []).append(d)
    return groups


def stream_prep(d, pr):
    """thermalization check, autocorrelation times, block length. Returns (start index, block length, info)."""
    N = len(d["ORIG_P"]); start = 0
    p = d["ORIG_P"]; q = N // 4
    def mean_err(x):
        t = tau_int(x); return x.mean(), x.std() * np.sqrt(2 * t / len(x))
    m1, e1 = mean_err(p[:q]); m2, e2 = mean_err(p[q:])
    therm_ok = abs(m1 - m2) < 3 * np.hypot(e1, e2)
    if not therm_ok:
        start = q
    taus = {}
    for e in ENS:
        taus[e + "_P"] = tau_int(d[f"{e}_P"][start:])
        lev = list(d["nsmear"]); shp = [tuple(x) for x in d["shapes"]]
        li = 0 if e == "CP" else lev.index(QLEV[float(d["beta"])][1])
        taus[e + "_op"] = tau_int(d[f"{e}_O"][start:, li, shp.index((2, 2))].mean(-1))
    tmax = max(taus.values()); n = N - start
    nb = int(np.clip(n // max(1, int(np.ceil(2 * tmax))), 8, 20))
    pr(f"    {d['src']:2s} {d['name']}: N={N} therm-check {'ok' if therm_ok else 'FAILED (first quarter dropped)'} "
       f"({m1:.5f} vs {m2:.5f})  tau_int max={tmax:.2f} [" + " ".join(f"{k}={v:.1f}" for k, v in taus.items()) + f"]  blocks={nb}")
    return start, nb, tmax


def block_sums(ds, preps, beta, basis, e, nb_override=None):
    """per-block (sum vector, count) over all streams; vector = [C(t) sums, op sums, W sums, plaq sum]."""
    out = []
    for d, (st, nb, _) in zip(ds, preps):
        if nb_override:
            nb = nb_override
        li, si = basis_index(d, beta, basis, e)
        O = d[f"{e}_O"][st:][:, li][:, :, si].astype(float)
        N, Lt = O.shape[0], O.shape[-1]; X = O.reshape(N, -1, Lt); Tm = Lt // 2
        W = d[f"{e}_W"][st:]; P = d[f"{e}_P"][st:]
        for idx in np.array_split(np.arange(N), nb):
            x = X[idx]
            C = np.stack([np.einsum("nit,njt->ij", x, np.roll(x, -t, -1)) / Lt for t in range(Tm + 1)])
            v = np.concatenate([C.ravel(), x.mean(-1).sum(0), W[idx].sum(0).ravel(), [P[idx].sum()]])
            out.append((v, len(idx)))
    return out, X.shape[1], Tm, W.shape[1:]


def jkw(f, blocks):
    """weighted delete-one-block jackknife. Returns (full, err, per-block estimates)."""
    tot = sum(v for v, _ in blocks); n = sum(c for _, c in blocks); B = len(blocks)
    full = f(tot / n)
    est = np.array([f((tot - v) / (n - c)) for v, c in blocks])
    with np.errstate(invalid="ignore"):
        err = np.sqrt((B - 1) / B * np.nansum((est - np.nanmean(est, 0)) ** 2, 0))
    return full, err, est


def make_fns(nop, Tm, wshape):
    nC = (Tm + 1) * nop * nop

    def corr(v):
        C = v[:nC].reshape(Tm + 1, nop, nop); m = v[nC:nC + nop]
        C = C - np.outer(m, m)[None]
        return 0.5 * (C + C.swapaxes(1, 2))
    wl = lambda v: v[nC + nop:nC + nop + wshape[0] * wshape[1]].reshape(wshape)

    def gevp_corr(v, t0, t1):
        C = corr(v)
        dn = np.sqrt(np.abs(np.diag(C[0]))) + 1e-300; Cn = C / np.outer(dn, dn)[None]
        if PRUNE and t0 > 0:                                     # deviation 1: prune to top C(0) modes first
            w0, Q0 = np.linalg.eigh(Cn[0]); V = Q0[:, ::-1][:, :PRUNE]
            Cn = np.einsum("ia,tij,jb->tab", V, Cn, V)
        w, Q = np.linalg.eigh(Cn[t0]); mk = w > 1e-4 * w.max()
        R = Q[:, mk] / np.sqrt(w[mk])
        lam, Y = np.linalg.eigh(R.T @ Cn[t1] @ R)
        u = R @ Y[:, -1]
        c = np.array([u @ Cn[t] @ u for t in range(Tm + 1)])
        return c / c[0]

    def meff(v, t0, t1):
        c = gevp_corr(v, t0, t1)
        with np.errstate(all="ignore"):
            return np.log(c[1:4] / c[2:5])                       # m(1), m(2), m(3)

    def sig_new(v):
        w = wl(v); R = np.arange(1, w.shape[0] + 1)
        with np.errstate(all="ignore"):
            V = np.log(w[:, 2] / w[:, 3])
        sel = R >= 2
        if not np.all(np.isfinite(V[sel])):
            return np.array([np.nan] * 3)
        A = np.stack([np.ones(sel.sum()), R[sel], -1.0 / R[sel]], 1)   # linear in (V0, sigma, e): exact LSQ
        return np.linalg.lstsq(A, V[sel], rcond=None)[0]

    def sig_old(v):
        w = wl(v); R = np.arange(1, w.shape[0] + 1); sel = R >= 2
        with np.errstate(all="ignore"):
            V = np.log(w[:, 1] / w[:, 2]) + np.pi / (12 * R)
        if not np.all(np.isfinite(V[sel])):
            return np.nan
        return np.polyfit(R[sel], V[sel], 1)[0]

    def pot(v):
        w = wl(v)
        with np.errstate(all="ignore"):
            return np.log(w[:, 2] / w[:, 3])
    return dict(gevp_corr=gevp_corr, meff=meff, sig_new=sig_new, sig_old=sig_old, pot=pot,
                old=lambda v: _old_m(gevp_corr, v), plaq=lambda v: v[-1])


def _old_m(gevp_corr, v):
    c = gevp_corr(v, 0, 1)
    with np.errstate(all="ignore"):
        return np.log(c[1] / c[2])


def plateau(me, mest):
    """pre-registered plateau rule on m(1), m(2). Returns (value, err, per-block est, flag)."""
    (m1, m2), (e1, e2) = me[0][:2], me[1][:2]
    d_est = mest[:, 0] - mest[:, 1]; B = len(d_est)
    de = np.sqrt((B - 1) / B * np.nansum((d_est - np.nanmean(d_est)) ** 2))
    ok2 = np.isfinite(m2) and e2 < 0.3 * abs(m2)
    if ok2 and abs(m1 - m2) < 1.5 * de:
        v, e, est, fl = m1, e1, mest[:, 0], "plateau"
    elif ok2:
        v, e, est, fl = m2, e2, mest[:, 1], "late"
    else:
        v, e, est, fl = m1, e1, mest[:, 0], "upper bound"
    if not np.isfinite(v) or e > 0.3 * abs(v):
        fl = "NOISY"
    return v, e, est, fl


def ratio_jk(a, ea, b, eb_est_a, est_b):
    """ratio a/b with jackknife from aligned per-block estimates."""
    r = a / b; est = eb_est_a / est_b; B = len(est)
    return r, np.sqrt((B - 1) / B * np.nansum((est - np.nanmean(est)) ** 2))


def verdict_ratio(r, e, box):
    c1 = abs(r - 1) < 2 * e; cb = abs(r - box) < 2 * e
    if c1 and not cb:
        return "GAP"
    if cb and not c1:
        return "GAPLESS"
    return "INCONCLUSIVE"


def analyze():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lines = []
    pr = lambda s="": (print(s), lines.append(s))
    groups = load_groups(); R = {}
    csv = ["beta,L,ensemble,basis,Nmeas,streams,tau_int_max,plaq,vortex_density,sigma_new_a2,sigma_new_err,coulomb_e,"
           "sigma_old_a2,sigma_old_err,m1,m1_err,m2,m2_err,m3,m3_err,m_quoted,m_quoted_err,flag,m_old_t0_0,m_old_err,"
           "a_sqs_unit,m_over_sqs_orig,m_over_sqs_err"]
    for (beta, L), ds in sorted(groups.items()):
        pr(f"=== beta={beta} L={L} streams={len(ds)} ({', '.join(d['src'] for d in ds)})")
        preps = [stream_prep(d, pr) for d in ds]
        gf = np.concatenate([d["gf"] for d in ds])
        pr(f"    MCG functional {gf[:,0].mean():.4f} -> {gf[:,1].mean():.4f} (min gain {np.min(gf[:,1]-gf[:,0]):.4f}), "
           f"iters {gf[:,2].mean():.0f}")
        bases = ("A", "B") if all(d["src"] == "Q2" for d in ds) else ("A",)
        if (beta, L) == (2.3, 12) or (beta, L) == (2.4, 12):
            bases = ("A", "B")                                     # B uses the new stream only
        for basis in bases:
            dsb = ds if basis == "A" else [d for d in ds if d["src"] == "Q2"]
            ppb = preps if basis == "A" else [p for d, p in zip(ds, preps) if d["src"] == "Q2"]
            if not dsb:
                continue
            for e in ENS:
                blocks, nop, Tm, wsh = block_sums(dsb, ppb, beta, basis, e)
                F = make_fns(nop, Tm, wsh)
                me = jkw(lambda v: F["meff"](v, 0, 1), blocks)
                mq, mqe, mest, flag = plateau(me, me[2])
                cg = jkw(lambda v: F["gevp_corr"](v, 0, 1), blocks)
                mo = jkw(F["old"], blocks)
                sn = jkw(F["sig_new"], blocks); so = jkw(F["sig_old"], blocks)
                pt = jkw(F["pot"], blocks)
                plq = jkw(F["plaq"], blocks)[0]
                N = sum(c for _, c in blocks)
                vd = np.mean(np.concatenate([d["vdens"] for d in dsb])) if e == "CP" else np.nan
                R[(beta, L, e, basis)] = dict(m=mq, me=mqe, mest=mest, flag=flag, meff=me, cg=cg, mo=mo, sn=sn, so=so,
                                              pot=pt, plaq=plq, N=N, vd=vd, nop=nop,
                                              tau=max(p[2] for p in ppb), streams=len(dsb), blocks=blocks, F=F)
                pr(f"  [{basis}] {e:4s} N={N} nop={nop} <P>={plq:.5f}"
                   + (f" vdens={vd:.4f}" if e == "CP" else "")
                   + f"  sigma(T3->4, free e)={sn[0][1]:.4f}({sn[1][1]:.4f}) e={sn[0][2]:.3f}"
                   f"  sigma(exploratory est)={so[0]:.4f}({so[1]:.4f})")
                pr(f"         m(1),m(2),m(3) = " + ", ".join(f"{a:.4f}({b:.4f})" for a, b in zip(me[0], me[1]))
                   + f"  quoted={mq:.4f}({mqe:.4f}) [{flag}]  exploratory-est m(t0=0,1->2)={mo[0]:.4f}({mo[1]:.4f})")
        pr()
    # units: a sqrt(sigma_ORIG) from 16^4, new estimator
    unit = {}
    for beta in (2.3, 2.4):
        k = (beta, 16, "ORIG", "A")
        if k in R:
            s, se = R[k]["sn"][0][1], R[k]["sn"][1][1]
            unit[beta] = (np.sqrt(s), se / (2 * np.sqrt(s)), np.sqrt(R[k]["sn"][2][:, 1]))
            pr(f"unit beta={beta}: a sqrt(sigma_ORIG) = {unit[beta][0]:.4f}({unit[beta][1]:.4f})  (lit {LIT_ASQS[beta]})  "
               f"<P> = {R[k]['plaq']:.5f} (lit {LIT_PLAQ[beta]}, |diff| {'<' if abs(R[k]['plaq']-LIT_PLAQ[beta])<0.003 else '>='} 0.003)")
    for (beta, L, e, basis), r in sorted(R.items()):
        u = unit.get(beta, (np.nan, np.nan))
        mu = r["m"] / u[0]; mue = abs(mu) * np.hypot(r["me"] / r["m"], u[1] / u[0])
        r["mu"], r["mue"] = mu, mue
        sn, so, me = r["sn"], r["so"], r["meff"]
        csv.append(f"{beta},{L},{e},{basis},{r['N']},{r['streams']},{r['tau']:.2f},{r['plaq']:.6f},{r['vd']:.5f},"
                   f"{sn[0][1]:.5f},{sn[1][1]:.5f},{sn[0][2]:.4f},{so[0]:.5f},{so[1]:.5f},"
                   + ",".join(f"{a:.5f},{b:.5f}" for a, b in zip(me[0], me[1]))
                   + f",{r['m']:.5f},{r['me']:.5f},{r['flag']},{r['mo'][0]:.5f},{r['mo'][1]:.5f},{u[0]:.5f},{mu:.4f},{mue:.4f}")
    # C0
    pr("\nC0 premise: sigma ratios (new estimator)")
    for (beta, L, e, basis), r in sorted(R.items()):
        if e == "ORIG" or basis != "A":
            continue
        o = R[(beta, L, "ORIG", "A")]
        rat, rate = ratio_jk(r["sn"][0][1], 0, o["sn"][0][1], r["sn"][2][:, 1], o["sn"][2][:, 1])
        ok = (rat < 0.15) if e == "VR" else (0.6 <= rat <= 1.4)
        pr(f"  beta={beta} L={L} sigma_{e}/sigma_ORIG = {rat:.3f}({rate:.3f})  {'PASS' if ok else 'FAIL'}")
        r["sigrat"] = (rat, rate, ok)
    # V test
    pr("\nV volume test: rho = m(16)/m(12), lattice units (independent ensembles)")
    for beta in (2.3, 2.4):
        for basis in ("A", "B"):
            for e in ENS:
                a, b = R.get((beta, 16, e, basis)), R.get((beta, 12, e, basis))
                if not a or not b:
                    continue
                rho = a["m"] / b["m"]; rhoe = rho * np.hypot(a["me"] / a["m"], b["me"] / b["m"])
                noisy = "NOISY" in (a["flag"], b["flag"])
                v = "NOISY" if noisy else verdict_ratio(rho, rhoe, 0.75)
                pr(f"  beta={beta} basis {basis} {e:4s}: {a['m']:.4f}({a['me']:.4f})[{a['flag']}] / {b['m']:.4f}({b['me']:.4f})"
                   f"[{b['flag']}] = {rho:.3f}({rhoe:.3f})  |rho-1|/err={abs(rho-1)/rhoe:.1f}  |rho-0.75|/err="
                   f"{abs(rho-0.75)/rhoe:.1f}  => {v}")
                R[(beta, "rho", e, basis)] = (rho, rhoe, v)
    from scipy.optimize import curve_fit
    pr("\nbeta 2.3 three-volume series (basis A): m(L) = A (12/L)^p")
    for e in ENS:
        pts = [(L, R[(2.3, L, e, "A")]) for L in (8, 12, 16) if (2.3, L, e, "A") in R]
        if len(pts) == 3:
            x = np.array([p[0] for p in pts], float); y = np.array([p[1]["m"] for p in pts]); ye = np.array([p[1]["me"] for p in pts])
            pp, cv = curve_fit(lambda L, A, p: A * (12 / L) ** p, x, y, p0=[y[1], 0.1], sigma=ye, absolute_sigma=True)
            chi = (((pp[0] * (12 / x) ** pp[1] - y) / ye) ** 2).sum()
            pr(f"  {e:4s}: p = {pp[1]:.3f}({np.sqrt(cv[1,1]):.3f})  chi2/dof={chi:.2f}  (0 = gap, 1 = box)")
    pr("\nC1: R1 = m_VR/m_ORIG at 16^4 (same configs, jackknife)")
    for beta in (2.3, 2.4):
        for basis in ("B", "A"):
            v, o = R.get((beta, 16, "VR", basis)), R.get((beta, 16, "ORIG", basis))
            if v and o:
                r1, r1e = ratio_jk(v["m"], 0, o["m"], v["mest"], o["mest"])
                pr(f"  beta={beta} basis {basis}: R1 = {r1:.3f}({r1e:.3f})   m/sqrt(sigma_ORIG): ORIG {o['mu']:.2f}({o['mue']:.2f})"
                   f" VR {v['mu']:.2f}({v['mue']:.2f}) CP {R[(beta,16,'CP',basis)]['mu']:.2f}({R[(beta,16,'CP',basis)]['mue']:.2f})")
    pr("\nC3: m_VR/sqrt(sigma_ORIG) at 16^4, beta 2.3 -> 2.4")
    for basis in ("B", "A"):
        a, b = R.get((2.3, 16, "VR", basis)), R.get((2.4, 16, "VR", basis))
        if a and b:
            d = b["mu"] - a["mu"]; de = np.hypot(a["mue"], b["mue"])
            pr(f"  basis {basis}: {a['mu']:.2f}({a['mue']:.2f}) -> {b['mu']:.2f}({b['mue']:.2f})  diff {d:+.2f}({de:.2f}) = {d/de:+.1f} sigma")
    pr("\nbin-size scan (VR, basis A, quoted-mass error vs blocks per stream)")
    groups2 = load_groups()
    for beta in (2.3, 2.4):
        for L in (12, 16):
            if (beta, L, "VR", "A") not in R:
                continue
            ds = groups2[(beta, L)]; pp = [(0, 0, 0)] * len(ds)
            out = []
            for nb in (4, 8, 12, 20, 40):
                blocks, nop, Tm, wsh = block_sums(ds, pp, beta, "A", "VR", nb_override=nb)
                F = make_fns(nop, Tm, wsh); me = jkw(lambda v: F["meff"](v, 0, 1), blocks)
                out.append(f"nb={nb}: m(1) err {me[1][0]:.4f}, m(2) err {me[1][1]:.4f}")
            pr(f"  beta={beta} L={L}: " + "; ".join(out))
    open(f"{OUT}/analyze.log", "w").write("\n".join(lines) + "\n")
    open(os.path.join(OUT, "summary.csv"), "w").write("\n".join(csv) + "\n")
    plots(R, unit, plt)


def plots(R, unit, plt):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e4e0", "grid.linewidth": 0.6, "axes.edgecolor": "#888", "lines.linewidth": 1.6})
    # 1. effective masses
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    for i, beta in enumerate((2.3, 2.4)):
        for j, L in enumerate((8, 12, 16)):
            a = ax[i, j]
            if (beta, L, "ORIG", "A") not in R:
                a.set_axis_off(); continue
            for k, e in enumerate(ENS):
                r = R[(beta, L, e, "A")]; t = np.arange(1, 4) + 0.08 * (k - 1)
                a.errorbar(t, r["meff"][0], r["meff"][1], fmt=MRK[e] + "-", color=COL[e], ms=6, capsize=2, label=e)
                if "B" in [kk[3] for kk in R if kk[:3] == (beta, L, e)]:
                    rb = R[(beta, L, e, "B")]
                    a.errorbar(t + 0.3, rb["meff"][0], rb["meff"][1], fmt=MRK[e], mfc="white", color=COL[e], ms=6, capsize=2)
            a.set_title(f"beta {beta}, {L}^4", fontsize=9); a.set_xticks([1, 2, 3])
            a.set_xticklabels(["1->2", "2->3", "3->4"])
            a.set_ylim(0, 2.0 if beta == 2.3 else 1.5)
            a.set_ylabel("a m_eff (GEVP t0=0, t=1)")
    ax[0, 0].legend(frameon=False, fontsize=8, title="filled: basis A, open: basis B", title_fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "effective-masses.png"), dpi=150); plt.close(fig)
    # 2. volume series
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for i, beta in enumerate((2.3, 2.4)):
        a = ax[i]
        for e in ENS:
            pts = [(L, R[(beta, L, e, "A")]) for L in (8, 12, 16) if (beta, L, e, "A") in R]
            x = np.array([p[0] for p in pts]); y = np.array([p[1]["m"] for p in pts]); ye = np.array([p[1]["me"] for p in pts])
            a.errorbar(x, y, ye, fmt=MRK[e] + "-", color=COL[e], ms=7, capsize=3, label=e)
            if e == "VR" and 12 in x:
                y12 = y[list(x).index(12)]; xx = np.linspace(8, 16, 50)
                a.plot(xx, y12 * 12 / xx, ":", color=COL[e], lw=1.2, label="VR if box-set (~1/L)")
        xx = np.linspace(8, 16, 50); a.plot(xx, 4 * np.sin(np.pi / xx), "--", color="#888", lw=1, label="free 2-gluon 4 sin(pi/L)")
        a.set_xlabel("L / a"); a.set_ylabel("a m_0++ (basis A, quoted)"); a.set_title(f"beta {beta}", fontsize=9)
        a.set_xticks([8, 12, 16]); a.set_ylim(0, 2.2)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "volume-series.png"), dpi=150); plt.close(fig)
    # 3. static potential at 16^4
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, beta in enumerate((2.3, 2.4)):
        a = ax[i]
        for e in ENS:
            r = R.get((beta, 16, e, "A"))
            if not r: continue
            Rr = np.arange(1, len(r["pot"][0]) + 1)
            a.errorbar(Rr, r["pot"][0], r["pot"][1], fmt=MRK[e], color=COL[e], ms=6, capsize=2, label=e)
            V0, s, c = r["sn"][0]; xx = np.linspace(1.6, Rr[-1], 60)
            a.plot(xx, V0 + s * xx - c / xx, "-", color=COL[e], lw=1.2)
        a.set_xlabel("R / a"); a.set_ylabel("a V(R) = ln W(R,3)/W(R,4)"); a.set_title(f"beta {beta}, 16^4", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "static-potential.png"), dpi=150); plt.close(fig)


def long24():
    """E1, E1' (R >= 4), E2 (T = 5->6, 6->7) and Creutz ratios chi(R,R) on the stored beta 2.4 Wilson loops, VR and CP
    over ORIG on the same configurations. Estimators and jackknife are those of gauge_copies.py (series 3 and the
    series 2 reanalysis), 20 blocks per stream. Added after referee review; not pre-registered."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gauge_copies", os.path.join(HERE, "gauge_copies.py"))
    gc = importlib.util.module_from_spec(spec); sys.modules["gauge_copies"] = gc; spec.loader.exec_module(gc)
    lines = []; pr = lambda s="": (print(s), lines.append(s))
    csv = ["beta,L,T,n_meas,ensemble,estimator,value,err,orig_value,orig_err,ratio_to_ORIG,ratio_err"]
    groups = load_groups()
    for L in (12, 16):
        ds = groups[(2.4, L)]
        assert all(stream_prep(d, pr)[:2] == (0, 20) for d in ds), "series 1 blocking is not 20 blocks from the start"
        nm = gc.est_names(L); k = len(nm); N = sum(len(d["VR_W"]) for d in ds)
        pr(f"=== beta=2.4 L={L} streams={len(ds)} N={N}")
        for e in ("VR", "CP"):
            full, err, _ = gc.jk_ratio_est(ds, 20, e)
            for i, n in enumerate(nm):
                pr(f"  {n:9s} {e:4s} {full[i]:+.5f}({err[i]:.5f})  ORIG {full[k+i]:+.5f}({err[k+i]:.5f})  "
                   f"ratio {full[2*k+i]:+.3f}({err[2*k+i]:.3f})")
                csv.append(f"2.4,{L},{L},{N},{e},{n},{full[i]:.5f},{err[i]:.5f},{full[k+i]:.5f},{err[k+i]:.5f},"
                           f"{full[2*k+i]:.4f},{err[2*k+i]:.4f}")
    open(f"{OUT}/long24.log", "w").write("\n".join(lines) + "\n")
    open(f"{OUT}/long24.csv", "w").write("\n".join(csv) + "\n")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if m == "selftest":
        selftest()
    elif m == "run":
        run(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
    elif m == "analyze":
        analyze()
    elif m == "long24":
        long24()
