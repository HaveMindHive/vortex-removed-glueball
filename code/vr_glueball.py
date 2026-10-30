#!/usr/bin/env python3
"""SU(2) Wilson-action ensembles; 0++ glueball and static potential on the original (ORIG), center-projected (CP) and
vortex-removed (VR) links in direct maximal center gauge (one gauge copy, over-relaxation, numba).

Modes:
  selftest
  time L T beta                          -> cost of one measurement
  run L T beta seed deadline_unix        -> runs/run_L{L}T{T}_b{beta}_s{seed}.npz (+ .ckpt); stops at the wall-clock deadline
  analyze                                -> runs/analyze.log, runs/summary.csv, PNGs in runs/
analyze() also reads ../data/series1_summary.csv (the beta 2.3 / 2.4 L^4 series) for the beta 2.4 volume test and plots.
"""
import sys, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")
S1CSV = os.path.join(HERE, "..", "data", "series1_summary.csv")   # series 1 results, read only
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
    def __init__(self, L, seed=0, T=None):
        self.L = L; self.T = T or L
        self.rng = np.random.default_rng(seed)
        U = self.rng.normal(size=(4, 4, self.T) + (L,) * 3)
        self.U = U / np.sqrt((U * U).sum(1, keepdims=True))
        g = np.indices((self.T,) + (L,) * 3).sum(0) % 2
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
    T = F.shape[2]; out = []
    lines = {(d, n): line(F, d, n) for d in (1, 2, 3) for n in range(1, L // 2 + 2)}
    for (r1, r2) in shapes(L):
        acc = np.zeros(T)
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


_NB = None


def _nb():
    """numba version of mcg's per-parity site update (same algorithm: exact local max = top eigenvector of M, then
    over-relaxation g -> g^omega; sites of one parity share no links, so in-place sequential = simultaneous)."""
    global _NB
    if _NB is None:
        from numba import njit

        @njit(cache=True)
        def qm(a0, a1, a2, a3, b0, b1, b2, b3):
            return (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                    a0 * b1 + b0 * a1 - (a2 * b3 - a3 * b2),
                    a0 * b2 + b0 * a2 - (a3 * b1 - a1 * b3),
                    a0 * b3 + b0 * a3 - (a1 * b2 - a2 * b1))

        @njit(cache=True)
        def top_eig(A, V):
            """cyclic Jacobi on a symmetric 4x4 (in place, no allocation); returns column index of the top eigenvalue."""
            for i in range(4):
                for j in range(4):
                    V[i, j] = 1.0 if i == j else 0.0
            for sweep in range(60):
                off = 0.0; dg = 0.0
                for i in range(4):
                    dg += A[i, i] * A[i, i]
                    for j in range(i + 1, 4):
                        off += A[i, j] * A[i, j]
                if off <= 1e-32 * dg:
                    break
                for p in range(3):
                    for q in range(p + 1, 4):
                        apq = A[p, q]
                        if apq == 0.0:
                            continue
                        th = (A[q, q] - A[p, p]) / (2.0 * apq)
                        t = (1.0 if th >= 0 else -1.0) / (abs(th) + np.sqrt(th * th + 1.0))
                        c = 1.0 / np.sqrt(t * t + 1.0); s = t * c
                        for k in range(4):
                            akp, akq = A[k, p], A[k, q]
                            A[k, p] = c * akp - s * akq; A[k, q] = s * akp + c * akq
                        for k in range(4):
                            apk, aqk = A[p, k], A[q, k]
                            A[p, k] = c * apk - s * aqk; A[q, k] = s * apk + c * aqk
                        for k in range(4):
                            vkp, vkq = V[k, p], V[k, q]
                            V[k, p] = c * vkp - s * vkq; V[k, q] = s * vkp + c * vkq
            b = 0
            for i in range(1, 4):
                if A[i, i] > A[b, b]:
                    b = i
            return b

        @njit(cache=True)
        def upd(U, parity, omega):
            T, L = U.shape[2], U.shape[3]
            M = np.zeros((4, 4)); V = np.zeros((4, 4)); ext = np.array([T, L, L, L]); m = np.zeros(4, np.int64)
            for t in range(T):
                for x in range(L):
                    for y in range(L):
                        for z in range(L):
                            if (t + x + y + z) % 2 != parity:
                                continue
                            M[:, :] = 0.0
                            for mu in range(4):
                                m[0] = t; m[1] = x; m[2] = y; m[3] = z; m[mu] = (m[mu] - 1) % ext[mu]
                                dv = (U[mu, 0, t, x, y, z], -U[mu, 1, t, x, y, z], -U[mu, 2, t, x, y, z], -U[mu, 3, t, x, y, z])
                                uv = (U[mu, 0, m[0], m[1], m[2], m[3]], U[mu, 1, m[0], m[1], m[2], m[3]],
                                      U[mu, 2, m[0], m[1], m[2], m[3]], U[mu, 3, m[0], m[1], m[2], m[3]])
                                for i in range(4):
                                    for j in range(4):
                                        M[i, j] += dv[i] * dv[j] + uv[i] * uv[j]
                            b = top_eig(M, V)
                            g0, g1, g2, g3 = V[0, b], V[1, b], V[2, b], V[3, b]
                            if g0 < 0:
                                g0, g1, g2, g3 = -g0, -g1, -g2, -g3
                            if omega != 1.0:
                                th = np.arccos(min(1.0, max(-1.0, g0))); sn = max(np.sin(th), 1e-300)
                                c, so = np.cos(omega * th), np.sin(omega * th) / sn
                                g0, g1, g2, g3 = c, so * g1, so * g2, so * g3
                                nn = np.sqrt(g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3)
                                g0, g1, g2, g3 = g0 / nn, g1 / nn, g2 / nn, g3 / nn
                            for mu in range(4):
                                m[0] = t; m[1] = x; m[2] = y; m[3] = z; m[mu] = (m[mu] - 1) % ext[mu]
                                r = qm(g0, g1, g2, g3, U[mu, 0, t, x, y, z], U[mu, 1, t, x, y, z], U[mu, 2, t, x, y, z], U[mu, 3, t, x, y, z])
                                for k in range(4):
                                    U[mu, k, t, x, y, z] = r[k]
                                r = qm(U[mu, 0, m[0], m[1], m[2], m[3]], U[mu, 1, m[0], m[1], m[2], m[3]],
                                       U[mu, 2, m[0], m[1], m[2], m[3]], U[mu, 3, m[0], m[1], m[2], m[3]], g0, -g1, -g2, -g3)
                                for k in range(4):
                                    U[mu, k, m[0], m[1], m[2], m[3]] = r[k]
        _NB = upd
    return _NB


def mcg_nb(U, omega=1.85, tol=1e-8, maxit=3000):
    """same as mcg (same stopping rule), numba site loop."""
    upd = _nb(); U = np.ascontiguousarray(U.copy()); F0 = mcg_functional(U); Fprev = F0
    for it in range(maxit):
        for parity in (0, 1):
            upd(U, parity, omega)
        U /= norm(U.swapaxes(0, 1))[:, None]
        Fn = mcg_functional(U)
        if abs(Fn - Fprev) < tol * Fn:
            break
        Fprev = Fn
    return U, F0, Fn, it + 1


def center_split(U, par):
    Ug, F0, F1, nit = mcg_nb(U)
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
        res[name] = (np.array(ops), Wl, P.mean(), (P < 0).mean() if name == "CP" else np.nan, P)
    # per-config check: MCG is a gauge transformation and every plaquette factorizes, P_orig = Z_P P_VR
    assert np.allclose(res["ORIG"][4], res["CP"][4] * res["VR"][4], atol=1e-8), "center factorization / gauge fixing broken"
    return res, gf


def run(L, T, beta, seed, deadline, ntherm=300, ckevery=10):
    import pickle, time
    fn = f"{OUT}/run_L{L}T{T}_b{beta}_s{seed}.npz"; ck = fn[:-4] + ".ckpt"
    if os.path.exists(ck):
        lat, D, vd, gfs, ts = pickle.load(open(ck, "rb")); print(f"resumed at {len(vd)} meas", flush=True)
    else:
        lat = Lattice(L, seed=seed, T=T)
        for _ in range(ntherm): lat.sweep(beta)
        print(f"thermalized {time.ctime()}", flush=True)
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
                     ts=np.array(ts), L=L, T=T, beta=beta, nsmear=nsmear(beta), shapes=np.array(shapes(L)))
            os.replace(fn + ".tmp.npz", fn)
            print(f"L={L} T={T} b={beta} s={seed} {i} P={np.mean(D['ORIG_P']):.5f} PVR={np.mean(D['VR_P']):.5f} "
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
    # numba MCG == numpy MCG (same algorithm), on a 4^3 x 6 lattice too
    for lt in (lat, Lattice(4, seed=3, T=6)):
        for _ in range(10): lt.sweep(2.5)
        A = mcg(lt.U, lt.par); B = mcg_nb(lt.U)
        print(f"MCG numpy vs numba: F {A[2]:.12f} vs {B[2]:.12f}, iters {A[3]} vs {B[3]}, max|dU| {np.abs(A[0]-B[0]).max():.2e}")
        assert abs(A[2] - B[2]) < 1e-9 and np.abs(A[0] - B[0]).max() < 1e-5
        r, gf = measure(lt, 2.5)
        assert r["ORIG"][0].shape[-1] == lt.T
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



# ---------------- analysis (pre-registered estimators) ----------------
QLEV = {2.3: (0, 2, 8), 2.4: (0, 4, 15), 2.5: (0, 8, 31)}   # basis A smearing levels (scaled with the string tension)
ASHAPES = [(1, 1), (2, 2), (3, 3), (4, 4), (1, 2), (2, 3), (3, 4)]
COL = {"ORIG": "#2a78d6", "VR": "#eb6834", "CP": "#1baf7a"}
MRK = {"ORIG": "o", "VR": "s", "CP": "^"}
LIT_ASQS = {2.3: 0.369, 2.4: 0.266, 2.5: 0.188}
LIT_PLAQ = {2.3: 0.6024, 2.4: 0.6300, 2.5: 0.6519}
VPAIR = {2.3: (16, 12), 2.5: (20, 16)}     # (large, small) L for the volume test; box value = small/large
NT = 6                                     # effective masses m(1..6) (m(3..6) report only)
PRUNE = 0                                  # GEVP at t0 = 0; pruning unused


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
    for fn in sorted(glob.glob(f"{OUT}/run_*.npz")):
        if fn.endswith(".tmp.npz"):
            continue
        d = dict(np.load(fn)); d["src"] = "S2"; d["name"] = os.path.basename(fn)
        groups.setdefault((float(d["beta"]), int(d["L"])), []).append(d)
    return groups


def q2_rows():
    import csv
    return {(float(r["beta"]), int(r["L"]), r["ensemble"], r["basis"]): r for r in csv.DictReader(open(S1CSV))}


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
        if PRUNE and t0 > 0:                                     # optional: prune to top C(0) modes first
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
            return np.log(c[1:NT + 1] / c[2:NT + 2])            # m(1) .. m(NT)

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


def late_plateau(me):
    """report only: first t with m(t), m(t+1) both rel err < 30% and within 1.5 sigma (jackknife of the difference)."""
    v, e, est = me; B = len(est)
    for t in range(len(v) - 1):
        d = est[:, t] - est[:, t + 1]; de = np.sqrt((B - 1) / B * np.nansum((d - np.nanmean(d)) ** 2))
        ok = all(np.isfinite(v[k]) and e[k] < 0.3 * abs(v[k]) for k in (t, t + 1))
        if ok and abs(v[t] - v[t + 1]) < 1.5 * de:
            return t + 1, v[t], e[t]
    return None


def analyze():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lines = []
    pr = lambda s="": (print(s), lines.append(s))
    groups = load_groups(); R = {}; S1 = q2_rows()
    csv = ["beta,L,T,ensemble,basis,Nmeas,streams,tau_int_max,plaq,vortex_density,sigma_new_a2,sigma_new_err,coulomb_e,"
           + ",".join(f"m{t},m{t}_err" for t in range(1, NT + 1))
           + ",m_quoted,m_quoted_err,flag,late_plateau_t,late_plateau_m,late_plateau_err,a_sqs_unit,m_over_sqs_orig,m_over_sqs_err"]
    for (beta, L), ds in sorted(groups.items()):
        T = int(ds[0]["T"])
        pr(f"=== beta={beta} L={L} T={T} streams={len(ds)}")
        preps = [stream_prep(d, pr) for d in ds]
        gf = np.concatenate([d["gf"] for d in ds])
        pr(f"    MCG functional {gf[:,0].mean():.4f} -> {gf[:,1].mean():.4f} (min gain {np.min(gf[:,1]-gf[:,0]):.4f}), "
           f"iters {gf[:,2].mean():.0f}")
        for basis in ("A", "B"):
            for e in ENS:
                blocks, nop, Tm, wsh = block_sums(ds, preps, beta, basis, e)
                F = make_fns(nop, Tm, wsh)
                me = jkw(lambda v: F["meff"](v, 0, 1), blocks)
                mq, mqe, mest, flag = plateau(me, me[2])
                sn = jkw(F["sig_new"], blocks); pt = jkw(F["pot"], blocks)
                plq = jkw(F["plaq"], blocks)[0]
                N = sum(c for _, c in blocks)
                vd = np.mean(np.concatenate([d["vdens"] for d in ds])) if e == "CP" else np.nan
                lp = late_plateau(me)
                R[(beta, L, e, basis)] = dict(m=mq, me=mqe, mest=mest, flag=flag, meff=me, sn=sn, pot=pt, plaq=plq, N=N,
                                              vd=vd, nop=nop, T=T, lp=lp, tau=max(p[2] for p in preps), streams=len(ds))
                pr(f"  [{basis}] {e:4s} N={N} nop={nop} <P>={plq:.5f}" + (f" vdens={vd:.4f}" if e == "CP" else "")
                   + f"  sigma(T3->4, free e)={sn[0][1]:.4f}({sn[1][1]:.4f}) e={sn[0][2]:.3f}")
                pr("         m(1..6) = " + ", ".join(f"{a:.4f}({b:.4f})" for a, b in zip(me[0], me[1]))
                   + f"  quoted={mq:.4f}({mqe:.4f}) [{flag}]  late plateau: "
                   + (f"t={lp[0]} m={lp[1]:.4f}({lp[2]:.4f})" if lp else "none"))
            if basis == "A":
                p = R[(beta, L, "ORIG", "A")]["plaq"]
                assert abs(p - LIT_PLAQ[beta]) < 0.003, f"plaquette {p} vs literature {LIT_PLAQ[beta]} at beta {beta}"
        pr()
    unit = {}
    for beta, (Lb, _) in VPAIR.items():
        k = (beta, Lb, "ORIG", "A")
        if k in R:
            sg, se = R[k]["sn"][0][1], R[k]["sn"][1][1]
            unit[beta] = (np.sqrt(sg), se / (2 * np.sqrt(sg)))
            pr(f"unit beta={beta}: a sqrt(sigma_ORIG) = {unit[beta][0]:.4f}({unit[beta][1]:.4f}) from L={Lb} (lit {LIT_ASQS[beta]})"
               f"  <P> = {R[k]['plaq']:.5f} (lit {LIT_PLAQ[beta]}, asserted within 0.003)")
    for (beta, L, e, basis), r in sorted(R.items()):
        u = unit.get(beta, (np.nan, np.nan))
        mu = r["m"] / u[0]; mue = abs(mu) * np.hypot(r["me"] / r["m"], u[1] / u[0])
        r["mu"], r["mue"] = mu, mue
        me, lp = r["meff"], r["lp"] or (np.nan, np.nan, np.nan)
        csv.append(f"{beta},{L},{r['T']},{e},{basis},{r['N']},{r['streams']},{r['tau']:.2f},{r['plaq']:.6f},{r['vd']:.5f},"
                   f"{r['sn'][0][1]:.5f},{r['sn'][1][1]:.5f},{r['sn'][0][2]:.4f},"
                   + ",".join(f"{a:.5f},{b:.5f}" for a, b in zip(me[0], me[1]))
                   + f",{r['m']:.5f},{r['me']:.5f},{r['flag']},{lp[0]},{lp[1]:.5f},{lp[2]:.5f},{u[0]:.5f},{mu:.4f},{mue:.4f}")
    pr("\nC0 premise: sigma ratios")
    for (beta, L, e, basis), r in sorted(R.items()):
        if e == "ORIG" or basis != "A":
            continue
        o = R[(beta, L, "ORIG", "A")]
        rat, rate = ratio_jk(r["sn"][0][1], 0, o["sn"][0][1], r["sn"][2][:, 1], o["sn"][2][:, 1])
        ok = (rat < 0.15) if e == "VR" else (0.6 <= rat <= 1.4)
        pr(f"  beta={beta} L={L} sigma_{e}/sigma_ORIG = {rat:.3f}({rate:.3f})  {'PASS' if ok else 'FAIL'}")
    pr("\nV volume tests: rho = m(L large)/m(L small), lattice units, independent ensembles")
    V = {}
    for beta, (Lb, Ls) in VPAIR.items():
        for basis in ("A", "B"):
            for e in ENS:
                a, b = R.get((beta, Lb, e, basis)), R.get((beta, Ls, e, basis))
                if not a or not b:
                    continue
                rho = a["m"] / b["m"]; rhoe = rho * np.hypot(a["me"] / a["m"], b["me"] / b["m"]); box = Ls / Lb
                v = "NOISY" if "NOISY" in (a["flag"], b["flag"]) else verdict_ratio(rho, rhoe, box)
                V[(beta, e, basis)] = (rho, rhoe, v, a["flag"], b["flag"])
                pr(f"  beta={beta} basis {basis} {e:4s}: {a['m']:.4f}({a['me']:.4f})[{a['flag']}] / {b['m']:.4f}({b['me']:.4f})"
                   f"[{b['flag']}] = {rho:.3f}({rhoe:.3f})  |rho-1|/err={abs(rho-1)/rhoe:.1f}  |rho-{box:.2f}|/err="
                   f"{abs(rho-box)/rhoe:.1f}  => {v}")
    def vcall(beta):
        if (beta, "VR", "A") not in V:
            return "MISSING"
        if V[(beta, "ORIG", "A")][2] != "GAP":
            return "NOT INTERPRETED (ORIG control " + V[(beta, "ORIG", "A")][2] + ")"
        return V[(beta, "VR", "A")][2]
    V23, V25, V24 = vcall(2.3), vcall(2.5), "GAP"
    pr(f"\nV23 = {V23}   V24 (S1, as recorded) = {V24}   V25 = {V25}")
    pr("\nC1: m_VR/m_ORIG at the largest box (same configs, jackknife)")
    for beta, (Lb, _) in VPAIR.items():
        for basis in ("A", "B"):
            v, o = R.get((beta, Lb, "VR", basis)), R.get((beta, Lb, "ORIG", basis))
            if v and o:
                r1, r1e = ratio_jk(v["m"], 0, o["m"], v["mest"], o["mest"])
                pr(f"  beta={beta} L={Lb} basis {basis}: R1 = {r1:.3f}({r1e:.3f})   m/sqrt(sigma_ORIG): ORIG {o['mu']:.2f}({o['mue']:.2f})"
                   f" VR {v['mu']:.2f}({v['mue']:.2f}) CP {R[(beta,Lb,'CP',basis)]['mu']:.2f}({R[(beta,Lb,'CP',basis)]['mue']:.2f})")
    q24 = S1[(2.4, 16, "VR", "A")]; m24, m24e = float(q24["m_over_sqs_orig"]), float(q24["m_over_sqs_err"])
    c3 = None
    if (2.5, 20, "VR", "A") in R:
        b = R[(2.5, 20, "VR", "A")]; d = b["mu"] - m24; de = np.hypot(m24e, b["mue"]); c3 = d / de
        pr(f"\nC3': m_VR/sqrt(sigma_ORIG) largest box, beta 2.4 (S1 16^4) {m24:.2f}({m24e:.2f}) -> 2.5 (20^4) "
           f"{b['mu']:.2f}({b['mue']:.2f}): diff {d:+.2f}({de:.2f}) = {c3:+.1f} sigma  "
           f"({'> 3 sigma DROP' if c3 < -3 else 'not a > 3 sigma drop'})")
    # combined verdict (pre-registered rule)
    both_plateau = (2.3, "VR", "A") in V and V[(2.3, "VR", "A")][3] == "plateau" and V[(2.3, "VR", "A")][4] == "plateau"
    if V25 == "GAPLESS" or (V23 == "GAPLESS" and both_plateau and V25 != "GAP"):
        verdict = "GAPLESS AFTER VORTEX REMOVAL"
    elif V25 == "GAP" and V24 == "GAP" and c3 is not None and c3 >= -3:
        verdict = ("GAP SURVIVES VORTEX REMOVAL (in this window)" if V23 != "GAPLESS"
                   else "GAP SURVIVES at beta >= 2.4, with a coarse-lattice exception at beta 2.3")
    else:
        verdict = "INCONCLUSIVE"
    pr(f"\nCOMBINED VERDICT: {verdict}")   # rule as written; does not apply the C0 premise check (see paper sec. 5)
    pr("\nbeta 2.3, S2 (T = 32) next to S1 (T = L), basis A, report only")
    for L in (12, 16):
        for e in ENS:
            r, q = R.get((2.3, L, e, "A")), S1.get((2.3, L, e, "A"))
            if r and q:
                pr(f"  L={L} {e:4s}: S2 m(1..3) " + ", ".join(f"{a:.3f}({b:.3f})" for a, b in zip(r["meff"][0][:3], r["meff"][1][:3]))
                   + f" quoted {r['m']:.3f}({r['me']:.3f})[{r['flag']}]   S1 " +
                   ", ".join(f"{float(q[f'm{t}']):.3f}({float(q[f'm{t}_err']):.3f})" for t in (1, 2, 3))
                   + f" quoted {float(q['m_quoted']):.3f}({float(q['m_quoted_err']):.3f})[{q['flag']}]")
    pr("\nbin-size scan (VR, basis A, error of m(1), m(2) vs blocks per stream)")
    for (beta, L), ds in sorted(groups.items()):
        pp = [(0, 0, 0)] * len(ds); out = []
        for nb in (4, 8, 12, 20, 40):
            blocks, nop, Tm, wsh = block_sums(ds, pp, beta, "A", "VR", nb_override=nb)
            F = make_fns(nop, Tm, wsh); me = jkw(lambda v: F["meff"](v, 0, 1), blocks)
            out.append(f"nb={nb}: {me[1][0]:.4f}, {me[1][1]:.4f}")
        pr(f"  beta={beta} L={L}: " + "; ".join(out))
    open(f"{OUT}/analyze.log", "w").write("\n".join(lines) + "\n")
    open(os.path.join(OUT, "summary.csv"), "w").write("\n".join(csv) + "\n")
    plots(R, unit, S1, plt)


def plots(R, unit, S1, plt):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e4e0", "grid.linewidth": 0.6, "axes.edgecolor": "#888", "lines.linewidth": 1.6})
    keys = [(2.3, 12), (2.3, 16), (2.5, 16), (2.5, 20)]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))
    for a, (beta, L) in zip(ax.ravel(), keys):
        if (beta, L, "ORIG", "A") not in R:
            a.set_axis_off(); continue
        for k, e in enumerate(ENS):
            r = R[(beta, L, e, "A")]; t = np.arange(1, NT + 1) + 0.08 * (k - 1)
            a.errorbar(t, r["meff"][0], r["meff"][1], fmt=MRK[e] + "-", color=COL[e], ms=5, capsize=2, label=e)
        a.set_title(f"beta {beta}, {L}^3 x {R[(beta, L, 'ORIG', 'A')]['T']}", fontsize=9)
        a.set_xticks(range(1, NT + 1)); a.set_xticklabels([f"{t}->{t+1}" for t in range(1, NT + 1)], fontsize=7)
        a.set_ylim(0, 1.8); a.set_ylabel("a m_eff (GEVP t0=0, t=1)")
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "effective-masses.png"), dpi=150); plt.close(fig)
    # quoted VR mass vs L at each beta (S1 points for 2.3 T=L and 2.4), lattice units, with the 1/L box line
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, beta in zip(ax, (2.3, 2.4, 2.5)):
        for e in ENS:
            s1 = [(L, float(S1[(beta, L, e, "A")]["m_quoted"]), float(S1[(beta, L, e, "A")]["m_quoted_err"]))
                  for L in (8, 12, 16) if (beta, L, e, "A") in S1]
            s2 = [(L, R[(beta, L, e, "A")]["m"], R[(beta, L, e, "A")]["me"]) for L in (12, 16, 20) if (beta, L, e, "A") in R]
            for pts, mfc, lab in ((s1, "white", "S1 (T = L)"), (s2, COL[e], "S2")):
                if pts:
                    x, y, ye = map(np.array, zip(*pts))
                    a.errorbar(x + (0.25 if mfc == "white" and beta == 2.3 else 0), y, ye, fmt=MRK[e] + "-", color=COL[e],
                               mfc=mfc, ms=6, capsize=2, label=f"{e} {lab}")
        xx = np.linspace(8, 20, 60); a.plot(xx, 4 * np.sin(np.pi / xx), "--", color="#888", lw=1, label="free 2-gluon")
        a.set_xlabel("L / a"); a.set_ylabel("a m_0++ (basis A, quoted)"); a.set_title(f"beta {beta}", fontsize=9)
        a.set_ylim(0, 1.6)
    ax[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "volume-series.png"), dpi=150); plt.close(fig)
    # VR mass in sqrt(sigma_ORIG) vs beta at the largest box
    fig, a = plt.subplots(figsize=(4.8, 3.6))
    for e in ENS:
        pts = [(2.3, float(S1[(2.3, 16, e, "A")]["m_over_sqs_orig"]), float(S1[(2.3, 16, e, "A")]["m_over_sqs_err"]), "white"),
               (2.4, float(S1[(2.4, 16, e, "A")]["m_over_sqs_orig"]), float(S1[(2.4, 16, e, "A")]["m_over_sqs_err"]), "white")]
        for beta, (Lb, _) in VPAIR.items():
            if (beta, Lb, e, "A") in R:
                pts.append((beta + 0.01, R[(beta, Lb, e, "A")]["mu"], R[(beta, Lb, e, "A")]["mue"], COL[e]))
        for x, y, ye, mfc in pts:
            a.errorbar(x, y, ye, fmt=MRK[e], color=COL[e], mfc=mfc, ms=6, capsize=2)
        a.plot([], [], MRK[e], color=COL[e], label=e)
    a.set_xlabel("beta (open: S1 16^4, filled: S2 largest box)"); a.set_ylabel("m_0++ / sqrt(sigma_ORIG)")
    a.legend(frameon=False, fontsize=8); a.set_ylim(0, 4.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mass-vs-beta.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if m == "selftest":
        selftest()
    elif m == "time":
        import time
        L, T, beta = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]); lat = Lattice(L, seed=9, T=T)
        t0 = time.time(); [lat.sweep(beta) for _ in range(5)]; t1 = time.time()
        for _ in range(20): lat.sweep(beta)
        t2 = time.time(); r, gf = measure(lat, beta); t3 = time.time()
        print(f"L={L} T={T} b={beta}: sweep {(t1-t0)/5:.2f}s  measure {t3-t2:.1f}s (MCG iters {gf[2]})  P={r['ORIG'][2]:.4f}")
    elif m == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
    elif m == "analyze":
        analyze()
