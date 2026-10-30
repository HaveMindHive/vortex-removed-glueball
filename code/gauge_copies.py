#!/usr/bin/env python3
"""Gauge-copy study of the residual VR string tension (SU(2), beta 2.5): every configuration gauge-fixed twice, paired.

Imports vr_glueball.py (unchanged). New here:
  - random gauge copies + the same over-relaxed direct maximal center gauge to convergence; the best functional among
    16 candidates (single copy + 15 random copies) is kept ("NEW"); the single copy alone is "OLD". NEW >= OLD per config.
    Optional simulated annealing (Metropolis, numba) is implemented and tested but was not used in production (NSA = 0).
  - Wilson loops W(R,T), R,T <= L/2, for ORIG, VR, CP with both gauge fixings, and basis-A glueball operators.
  - string-tension estimators E1 (T 3->4), E1' (R = 4..L/2), E2 (T 5->6, 6->7), E3 (Creutz ratios chi(R,R)).
Modes: selftest | time L T beta | run L T beta seed deadline_unix | reanalyze_series2 | analyze
"""
import sys, os, glob, importlib.util, pickle, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_gauge_copies")
os.makedirs(OUT, exist_ok=True)
_spec = importlib.util.spec_from_file_location("vr_glueball", os.path.join(HERE, "vr_glueball.py"))
vg = importlib.util.module_from_spec(_spec); sys.modules["vr_glueball"] = vg; _spec.loader.exec_module(vg)   # registered: checkpoints pickle the Lattice class
mul, dag, norm, sh = vg.mul, vg.dag, vg.norm, vg.sh

NCOPY = 15         # random gauge copies + OR, plus the OLD copy as a candidate (16 candidates)
NSA = 0            # SA sweeps per copy (0: not used in production)
TSA = (1.5, 0.01)  # SA temperature, geometric schedule
NSKIP4 = 10        # compound sweeps between measurements (S2: 2)
VARS = ("o", "n")  # OLD, NEW gauge fixing

_SA = None


def _sa():
    global _SA
    if _SA is None:
        from numba import njit

        @njit(cache=False)
        def seed(s):
            np.random.seed(s)

        @njit(cache=False)
        def sweep(U, parity, temp, eps):
            """Metropolis on the site gauge g (unit quaternion), local functional g^T M g (as the MCG update), 2 hits."""
            T, L = U.shape[2], U.shape[3]
            M = np.zeros((4, 4)); ext = np.array([T, L, L, L]); m = np.zeros(4, np.int64); acc = 0; tot = 0
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
                            g0, g1, g2, g3 = 1.0, 0.0, 0.0, 0.0
                            F = M[0, 0]
                            for hit in range(2):
                                a = eps * np.random.random()
                                n1, n2, n3 = np.random.normal(), np.random.normal(), np.random.normal()
                                nn = np.sqrt(n1 * n1 + n2 * n2 + n3 * n3)
                                c, s = np.cos(a), np.sin(a) / nn
                                r0, r1, r2, r3 = c, s * n1, s * n2, s * n3
                                h0 = r0 * g0 - r1 * g1 - r2 * g2 - r3 * g3
                                h1 = r0 * g1 + g0 * r1 - (r2 * g3 - r3 * g2)
                                h2 = r0 * g2 + g0 * r2 - (r3 * g1 - r1 * g3)
                                h3 = r0 * g3 + g0 * r3 - (r1 * g2 - r2 * g1)
                                h = (h0, h1, h2, h3); Fn = 0.0
                                for i in range(4):
                                    for j in range(4):
                                        Fn += h[i] * M[i, j] * h[j]
                                tot += 1
                                if Fn >= F or np.random.random() < np.exp((Fn - F) / temp):
                                    g0, g1, g2, g3, F = h0, h1, h2, h3, Fn; acc += 1
                            for mu in range(4):
                                m[0] = t; m[1] = x; m[2] = y; m[3] = z; m[mu] = (m[mu] - 1) % ext[mu]
                                a0, a1, a2, a3 = U[mu, 0, t, x, y, z], U[mu, 1, t, x, y, z], U[mu, 2, t, x, y, z], U[mu, 3, t, x, y, z]
                                U[mu, 0, t, x, y, z] = g0 * a0 - g1 * a1 - g2 * a2 - g3 * a3
                                U[mu, 1, t, x, y, z] = g0 * a1 + a0 * g1 - (g2 * a3 - g3 * a2)
                                U[mu, 2, t, x, y, z] = g0 * a2 + a0 * g2 - (g3 * a1 - g1 * a3)
                                U[mu, 3, t, x, y, z] = g0 * a3 + a0 * g3 - (g1 * a2 - g2 * a1)
                                b0, b1, b2, b3 = U[mu, 0, m[0], m[1], m[2], m[3]], U[mu, 1, m[0], m[1], m[2], m[3]], \
                                    U[mu, 2, m[0], m[1], m[2], m[3]], U[mu, 3, m[0], m[1], m[2], m[3]]
                                c0, c1, c2, c3 = g0, -g1, -g2, -g3
                                U[mu, 0, m[0], m[1], m[2], m[3]] = b0 * c0 - b1 * c1 - b2 * c2 - b3 * c3
                                U[mu, 1, m[0], m[1], m[2], m[3]] = b0 * c1 + c0 * b1 - (b2 * c3 - b3 * c2)
                                U[mu, 2, m[0], m[1], m[2], m[3]] = b0 * c2 + c0 * b2 - (b3 * c1 - b1 * c3)
                                U[mu, 3, m[0], m[1], m[2], m[3]] = b0 * c3 + c0 * b3 - (b1 * c2 - b2 * c1)
            return acc, tot
        _SA = (seed, sweep)
    return _SA


def random_gauge(U, rng):
    G = rng.normal(size=U.shape[1:]); G /= norm(G)
    return np.stack([mul(mul(G, U[mu]), dag(sh(G, mu, 1))) for mu in range(4)])


def anneal(U, nsa=NSA, tsa=TSA, eps=1.0):
    seed, sweep = _sa(); U = np.ascontiguousarray(U.copy())
    for temp in np.geomspace(tsa[0], tsa[1], nsa):
        for parity in (0, 1):
            sweep(U, parity, temp, eps)
        U /= norm(U.swapaxes(0, 1))[:, None]
    return U


def gauge_fix(U, rng, ncopy=NCOPY, nsa=NSA):
    """OLD = S2 (one copy, OR from the config as given). NEW = best functional over {OLD, ncopy x (random gauge, [SA], OR)}.
    Returns OLD, NEW, [F0, it_old, F_old, F_copy1..], and every candidate's gauge-fixed links (copy 0 = OLD)."""
    Uo, F0, Fo, it = vg.mcg_nb(U)
    cands, Fs = [Uo], [Fo]
    for c in range(ncopy):
        Ur = random_gauge(U, rng)
        Uc, _, Fc, _ = vg.mcg_nb(anneal(Ur, nsa) if nsa else Ur)
        cands.append(Uc); Fs.append(Fc)
    return Uo, cands[int(np.argmax(Fs))], np.array([F0, it] + Fs), cands


def split(Ug):
    Z = np.sign(Ug[:, 0]); CP = np.zeros_like(Ug); CP[:, 0] = Z
    return Ug * Z[:, None], CP


def measure4(lat, beta, rng, ncopy=NCOPY, nsa=NSA):
    L = lat.L; lev = vg.QLEV[beta]; ns = vg.nsmear(beta); wlev = ns[1]          # Wilson loops at S2's level index 1
    assert wlev == lev[1]
    Uo, Un, gf, cands = gauge_fix(lat.U, rng, ncopy, nsa)
    VRo, CPo = split(Uo); VRn, CPn = split(Un)
    res = {}
    # every candidate copy: VR loops at the Wilson level, CP loops unsmeared (dose-response, report only)
    cw = []
    for Uc in cands:
        VRc, CPc = split(Uc)
        cw.append((vg.wilson(np.concatenate([VRc[:1], vg.ape(VRc[1:], wlev)]), VRc[0], L, L // 2), vg.wilson(CPc, CPc[0], L, L // 2)))
    res["VRc"] = (np.zeros(0, np.float32), np.array([c[0] for c in cw]), 0.0, None)
    res["CPc"] = (np.zeros(0, np.float32), np.array([c[1] for c in cw]), 0.0, None)
    for name, U in (("ORIG", lat.U), ("VRo", VRo), ("VRn", VRn), ("CPo", CPo), ("CPn", CPn)):
        levels = (0,) if name.startswith("CP") else lev
        ops, Us, done, Wl = [], U[1:].copy(), 0, None
        for n in levels:
            Us = vg.ape(Us, n - done); done = n
            F = np.concatenate([U[:1], Us])
            ops.append(vg.loop_ops(F, L))
            if n == wlev or name.startswith("CP"):
                Wl = vg.wilson(F, U[0], L, L // 2)
        res[name] = (np.array(ops, np.float32), Wl, vg.plaq_field(U).mean(), vg.plaq_field(U))
    assert np.allclose(res["VRc"][1][0], res["VRo"][1]) and np.allclose(res["CPc"][1][0], res["CPo"][1])
    for v in ("o", "n"):   # MCG is a gauge transformation; center factorization per plaquette
        assert np.allclose(res["ORIG"][3], res["CP" + v][3] * res["VR" + v][3], atol=1e-8), "center factorization broken"
    return {k: r[:3] for k, r in res.items()}, gf


def run(L, T, beta, seed, deadline, ntherm=300, ckevery=5):
    fn = f"{OUT}/run_L{L}T{T}_b{beta}_s{seed}.npz"; ck = fn[:-4] + ".ckpt"
    if os.path.exists(ck):
        lat, D, gfs, ts, rng = pickle.load(open(ck, "rb")); print(f"resumed at {len(ts)} meas", flush=True)
    else:
        lat = vg.Lattice(L, seed=seed, T=T); rng = np.random.default_rng(seed + 10000)
        _sa()[0](seed)
        for _ in range(ntherm): lat.sweep(beta)
        print(f"thermalized {time.ctime()}", flush=True)
        D, gfs, ts = {}, [], []
    _sa()[0](seed + 20000 + len(ts))
    while time.time() < deadline:
        for _ in range(NSKIP4): lat.sweep(beta)
        assert np.allclose(norm(lat.U.swapaxes(0, 1)), 1, atol=1e-10), "links left SU(2)"
        res, gf = measure4(lat, beta, rng)
        assert gf[2] >= gf[0] and gf[2:].max() >= gf[2], "gauge fixing broken"
        for e, (O, W, P) in res.items():
            for k, x in (("O", O), ("W", W), ("P", P)):
                D.setdefault(f"{e}_{k}", []).append(x)
        gfs.append(gf); ts.append(time.time()); i = len(ts)
        if i % ckevery == 0 or time.time() >= deadline:
            pickle.dump((lat, D, gfs, ts, rng), open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
            np.savez(fn + ".tmp.npz", **{k: np.array(v) for k, v in D.items()}, gf=np.array(gfs), ts=np.array(ts), L=L, T=T,
                     beta=beta, nsmear=np.array(vg.QLEV[beta]), shapes=np.array(vg.shapes(L)))
            os.replace(fn + ".tmp.npz", fn)
            g = np.array(gfs)
            print(f"L={L} T={T} b={beta} s={seed} {i} P={np.mean(D['ORIG_P']):.5f} F_old={g[:,2].mean():.5f} "
                  f"F_new={g[:,2:].max(1).mean():.5f} new>old {np.mean(g[:,3:].max(1) > g[:,2] + 1e-9):.2f} "
                  f"dt={ts[-1]-ts[-2] if i>1 else 0:.1f}s", flush=True)
    print("deadline reached", flush=True)


def selftest():
    rng = np.random.default_rng(5); _sa()[0](5)
    lat = vg.Lattice(4, seed=2, T=6)
    for _ in range(20): lat.sweep(2.5)
    Ug = random_gauge(lat.U, rng)
    assert np.allclose(vg.plaq_field(Ug), vg.plaq_field(lat.U), atol=1e-10), "random gauge not a gauge transformation"
    Ua = anneal(Ug, 100)
    assert np.allclose(vg.plaq_field(Ua), vg.plaq_field(lat.U), atol=1e-10), "SA not a gauge transformation"
    print(f"SA: functional {vg.mcg_functional(Ug):.4f} -> {vg.mcg_functional(Ua):.4f}")
    assert vg.mcg_functional(Ua) > 0.6
    # cold SA (T -> 0) never lowers the functional
    U1 = anneal(Ua, 5, (1e-9, 1e-9)); assert vg.mcg_functional(U1) >= vg.mcg_functional(Ua) - 1e-12
    res, gf = measure4(lat, 2.5, rng, ncopy=2, nsa=50)
    assert np.allclose(res["VRc"][1][int(np.argmax(gf[2:]))], res["VRn"][1])
    print("gf [F0, it, F_old, F_copies]:", np.round(gf, 5))
    assert gf[2:].max() >= gf[2]
    assert res["VRo"][1].shape == (2, 2) and res["ORIG"][0].shape == (3, len(vg.shapes(4)), 6)
    # old variant reproduces S2's measure (same gauge fixing, same loops) on the same config
    r3, _ = vg.measure(lat, 2.5)
    assert np.allclose(r3["VR"][1], res["VRo"][1]) and np.allclose(r3["ORIG"][1], res["ORIG"][1])
    assert np.allclose(r3["VR"][0][[0, 1, 3]], res["VRo"][0], rtol=1e-5, atol=1e-4)
    # estimators on synthetic area-law loops: sigma recovered by E1, E2, E3
    R, Tt = np.meshgrid(np.arange(1, 9), np.arange(1, 9), indexing="ij")
    W = np.exp(-0.05 * R * Tt - 0.1 * (R + Tt) + 0.2 * Tt / R)
    for f in (lambda w: sig_T(w, 3), lambda w: sig_T(w, 6), lambda w: sig_T(w, 3, 4)):
        assert abs(f(W)[1] - 0.05) < 1e-9, f(W)
    assert np.allclose(creutz(np.exp(-0.05 * R * Tt - 0.1 * (R + Tt)))[1:], 0.05)
    print("selftest OK")


# ---------------- estimators ----------------
def sig_T(w, T, Rmin=2):
    """V(R) = ln W(R,T)/W(R,T+1), fit V0 + sigma R - e/R on R = Rmin..L/2 (S1/S2 estimator at T = 3)."""
    R = np.arange(1, w.shape[0] + 1)
    if T + 1 > w.shape[1]:
        return np.full(3, np.nan)
    with np.errstate(all="ignore"):
        V = np.log(w[:, T - 1] / w[:, T])
    sel = R >= Rmin
    if not np.all(np.isfinite(V[sel])):
        return np.full(3, np.nan)
    A = np.stack([np.ones(sel.sum()), R[sel], -1.0 / R[sel]], 1)
    return np.linalg.lstsq(A, V[sel], rcond=None)[0]


def creutz(w):
    """chi(R,R) for R = 1..L/2 (chi(1) = -ln W(1,1))."""
    d = np.diag(w); out = [-np.log(d[0])]
    with np.errstate(all="ignore"):
        for R in range(2, w.shape[0] + 1):
            out.append(-np.log(w[R - 1, R - 1] * w[R - 2, R - 2] / (w[R - 1, R - 2] * w[R - 2, R - 1])))
    return np.array(out)


def est_vec(w):
    """vector of string-tension estimates: [E1, E1'(R>=4), E2 T=5, E2 T=6, chi(2..L/2)]."""
    return np.concatenate([[sig_T(w, 3)[1], sig_T(w, 3, 4)[1], sig_T(w, 5)[1], sig_T(w, 6)[1]], creutz(w)[1:]])


def est_names(L):
    return ["E1", "E1'", "E2_T5", "E2_T6"] + [f"chi({R},{R})" for R in range(2, L // 2 + 1)]


def blocks_W(ds, nb_per, e):
    """per-block (sum of W, count) for ensemble key e over streams (W only)."""
    out = []
    for d in ds:
        W = d[f"{e}_W"]
        for idx in np.array_split(np.arange(len(W)), nb_per):
            out.append((W[idx].sum(0), len(idx)))
    return out


def jk_ratio_est(ds, nb, e, eo="ORIG"):
    """jackknife of est_vec(e), est_vec(ORIG), ratio e/ORIG (same blocks)."""
    ba, bo = blocks_W(ds, nb, e), blocks_W(ds, nb, eo)
    def f(v):
        k = v.size // 2; a, o = v[:k].reshape(ba[0][0].shape), v[k:].reshape(ba[0][0].shape)
        ea, eo_ = est_vec(a), est_vec(o)
        return np.concatenate([ea, eo_, ea / eo_])
    bl = [(np.concatenate([a.ravel(), o.ravel()]), c) for (a, c), (o, _) in zip(ba, bo)]
    full, err, est = vg.jkw(f, bl)
    return full, err, est


def reanalyze_series2():
    """E1, E1', E2, E3 on the stored S2 Wilson loops (OLD gauge fixing, S2 configs). Report only."""
    lines = []; pr = lambda s="": (print(s), lines.append(s))
    for (beta, L), ds in sorted(vg.load_groups().items()):
        nm = est_names(L); k = len(nm)
        pr(f"=== S2 stored loops beta={beta} L={L} streams={len(ds)} N={sum(len(d['VR_W']) for d in ds)}")
        for e in ("VR", "CP"):
            full, err, _ = jk_ratio_est(ds, 20, e)
            for i, n in enumerate(nm):
                pr(f"  {n:9s} {e:4s} {full[i]:+.5f}({err[i]:.5f})  ORIG {full[k+i]:+.5f}({err[k+i]:.5f})  "
                   f"ratio {full[2*k+i]:+.3f}({err[2*k+i]:.3f})")
    open(f"{OUT}/series2-reanalysis.log", "w").write("\n".join(lines) + "\n")


def view(d, v):
    """S2-format dict for gauge-fixing variant v ('o' or 'n')."""
    out = {k: d[k] for k in ("L", "T", "beta", "nsmear", "shapes", "gf")}
    out.update({f"ORIG_{k}": d[f"ORIG_{k}"] for k in "OWP"})
    out.update({f"VR_{k}": d[f"VR{v}_{k}"] for k in "OWP"}); out.update({f"CP_{k}": d[f"CP{v}_{k}"] for k in "OWP"})
    out["src"], out["name"] = "S3", d["name"] + f"[{v}]"
    return out


def analyze():
    """Pre-registered gauge-copy analysis."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lines = []; pr = lambda s="": (print(s), lines.append(s))
    groups = {}
    for fn in sorted(glob.glob(f"{OUT}/run_*.npz")):
        if fn.endswith(".tmp.npz"):
            continue
        d = dict(np.load(fn)); d["name"] = os.path.basename(fn)
        groups.setdefault((float(d["beta"]), int(d["L"]), int(d["T"])), []).append(d)
    csv = ["beta,L,T,gf,quantity,value,err"]; S = {}
    for (beta, L, T), ds in sorted(groups.items()):
        pr(f"=== beta={beta} L={L} T={T} streams={len(ds)} N={sum(len(d['ts']) for d in ds)}")
        g = np.concatenate([d["gf"] for d in ds]); Fo, Fn = g[:, 2], g[:, 2:].max(1)
        pr(f"  MCG functional: start {g[:,0].mean():.4f}; OLD {Fo.mean():.5f}; NEW {Fn.mean():.5f}; NEW-OLD mean "
           f"{(Fn-Fo).mean():.5f} (max {(Fn-Fo).max():.5f}); configs with a better copy {np.mean(Fn > Fo + 1e-9):.2f}; "
           f"spread of random copies (max-min) {np.mean(g[:,3:].max(1)-g[:,3:].min(1)):.5f}")
        csv += [f"{beta},{L},{T},o,F_mcg,{Fo.mean():.6f},{Fo.std()/np.sqrt(len(Fo)):.6f}",
                f"{beta},{L},{T},n,F_mcg,{Fn.mean():.6f},{Fn.std()/np.sqrt(len(Fn)):.6f}"]
        vo = [view(d, "o") for d in ds]; vn = [view(d, "n") for d in ds]
        preps = [vg.stream_prep(d, pr) for d in vo]
        nb = min(p[1] for p in preps)
        # string tension estimators, ratios to ORIG, and NEW - OLD paired
        nm = est_names(L); k = len(nm)
        for e in ("VR", "CP"):
            ro = jk_ratio_est(ds, nb, e + "o"); rn = jk_ratio_est(ds, nb, e + "n")
            dif = rn[2] - ro[2]; B = len(dif); dfe = np.sqrt((B - 1) / B * np.nansum((dif - np.nanmean(dif, 0)) ** 2, 0))
            for i, n in enumerate(nm):
                pr(f"  {n:9s} {e}: OLD {ro[0][i]:+.5f}({ro[1][i]:.5f}) ratio {ro[0][2*k+i]:+.3f}({ro[1][2*k+i]:.3f}) | NEW "
                   f"{rn[0][i]:+.5f}({rn[1][i]:.5f}) ratio {rn[0][2*k+i]:+.3f}({rn[1][2*k+i]:.3f}) | NEW-OLD ratio "
                   f"{rn[0][2*k+i]-ro[0][2*k+i]:+.3f}({dfe[2*k+i]:.3f}) | ORIG {ro[0][k+i]:+.5f}({ro[1][k+i]:.5f})")
                for v, r in (("o", ro), ("n", rn)):
                    csv.append(f"{beta},{L},{T},{v},{e}/ORIG {n},{r[0][2*k+i]:.5f},{r[1][2*k+i]:.5f}")
                S[(beta, L, e, n)] = (ro[0][2*k+i], ro[1][2*k+i], rn[0][2*k+i], rn[1][2*k+i], rn[0][2*k+i]-ro[0][2*k+i], dfe[2*k+i],
                                      ro[0][:k], ro[1][:k], rn[0][:k], rn[1][:k], ro[0][k:2*k], ro[1][k:2*k])
        # dose-response (report only): copies ranked by functional within each config, E1 ratio per rank
        Fc = np.concatenate([d["gf"][:, 2:] for d in ds]); order = np.argsort(-Fc, 1)
        for e in ("VR", "CP"):
            Wc = np.concatenate([d[f"{e}c_W"] for d in ds]); Wr = np.take_along_axis(Wc, order[:, :, None, None], 1)
            Wo = np.concatenate([d["ORIG_W"] for d in ds]); nr = Wr.shape[1]; out = []
            for rk in range(nr):
                bl = [(np.concatenate([Wr[idx, rk].sum(0).ravel(), Wo[idx].sum(0).ravel()]), len(idx))
                      for idx in np.array_split(np.arange(len(Wo)), nb * len(ds))]
                f = lambda v: np.array([sig_T(v[:v.size // 2].reshape(Wo.shape[1:]), 3)[1] / sig_T(v[v.size // 2:].reshape(Wo.shape[1:]), 3)[1]])
                full, err, _ = vg.jkw(f, bl); out.append((full[0], err[0]))
                csv.append(f"{beta},{L},{T},rank{rk},{e}/ORIG E1 by functional rank,{full[0]:.5f},{err[0]:.5f}")
            S[(beta, L, e, "rank")] = (np.array(out), np.sort(Fc, 1)[:, ::-1].mean(0))
            pr(f"  E1 {e}/ORIG by functional rank (0 = best of {nr}; mean F per rank {np.sort(Fc,1)[:,::-1].mean(0)[0]:.5f} .. "
               f"{np.sort(Fc,1)[:,::-1].mean(0)[-1]:.5f}): " + ", ".join(f"{a:.3f}({b:.3f})" for a, b in out))
        # glueball masses, basis A, S2 estimator and plateau rule, OLD and NEW on the same blocks
        M = {}
        for v, vs in (("o", vo), ("n", vn)):
            for e in ("ORIG", "VR"):
                blocks, nop, Tm, wsh = vg.block_sums(vs, [(p[0], nb, p[2]) for p in preps], beta, "A", e)
                F = vg.make_fns(nop, Tm, wsh)
                me = vg.jkw(lambda x: F["meff"](x, 0, 1), blocks)
                mq, mqe, mest, flag = vg.plateau(me, me[2])
                M[(e, v)] = (mq, mqe, mest, flag, me)
                pr(f"  [A] {e:4s} {v}: m(1..4) = " + ", ".join(f"{a:.4f}({b:.4f})" for a, b in zip(me[0][:4], me[1][:4]))
                   + f"  quoted {mq:.4f}({mqe:.4f}) [{flag}]")
                csv.append(f"{beta},{L},{T},{v},m_{e}_quoted[{flag}],{mq:.5f},{mqe:.5f}")
        a, b = M[("VR", "n")], M[("VR", "o")]
        d = a[2] - b[2]; B = len(d); de = np.sqrt((B - 1) / B * np.nansum((d - np.nanmean(d)) ** 2))
        pr(f"  VR mass NEW - OLD (paired, quoted values) = {a[0]-b[0]:+.4f}({de:.4f}) = {(a[0]-b[0])/de:+.1f} sigma")
        d1 = a[4][2][:, 0] - b[4][2][:, 0]; d1e = np.sqrt((B - 1) / B * np.nansum((d1 - np.nanmean(d1)) ** 2))
        pr(f"  VR m(1) NEW - OLD (paired) = {a[4][0][0]-b[4][0][0]:+.4f}({d1e:.4f})")
        csv.append(f"{beta},{L},{T},n-o,m_VR_quoted_diff,{a[0]-b[0]:.5f},{de:.5f}")
        S[(beta, L, "mass")] = (b[0], b[1], b[3], a[0], a[1], a[3], a[0] - b[0], de)
        S[(beta, L, "F")] = (Fo, Fn)
        pr()
    open(f"{OUT}/analyze.log", "w").write("\n".join(lines) + "\n")
    open(os.path.join(OUT, "summary.csv"), "w").write("\n".join(csv) + "\n")
    pickle.dump(S, open(f"{OUT}/summary.pkl", "wb"))
    plot(S, plt)
    return S


def plot(S, plt):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e4e0", "grid.linewidth": 0.6})
    keys = sorted({(k[0], k[1]) for k in S if len(k) == 4 and k[3] != "rank"})
    fig, ax = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.8), squeeze=False)
    for a, (beta, L) in zip(ax[0], keys):
        nm = est_names(L)
        for j, (v, col, mk) in enumerate((("o", "#eb6834", "s"), ("n", "#2a78d6", "o"))):
            y = [S[(beta, L, "VR", n)][0 if v == "o" else 2] for n in nm]
            ye = [S[(beta, L, "VR", n)][1 if v == "o" else 3] for n in nm]
            a.errorbar(np.arange(len(nm)) + 0.12 * (j - 0.5), y, ye, fmt=mk, color=col, ms=5, capsize=2,
                       label="VR, " + ("OLD gauge fixing (S2)" if v == "o" else "NEW (best of 16 copies)"))
        a.axhline(0.15, color="#888", ls="--", lw=1, label="C0 bound 0.15"); a.axhline(0, color="#444", lw=0.8)
        a.set_xticks(range(len(nm))); a.set_xticklabels(nm, rotation=45, fontsize=7)
        a.set_ylabel("sigma_VR / sigma_ORIG"); a.set_title(f"beta {beta}, L {L}", fontsize=9)
    ax[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sigma-ratios.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if m == "selftest":
        selftest()
    elif m == "time":
        L, T, beta = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]); lat = vg.Lattice(L, seed=9, T=T)
        rng = np.random.default_rng(1); _sa()[0](1); _sa()[1](np.zeros((4, 4, 2, 2, 2, 2)) + 0.5, 0, 1.0, 1.0)
        t0 = time.time(); [lat.sweep(beta) for _ in range(5)]; t1 = time.time()
        for _ in range(25): lat.sweep(beta)
        t2 = time.time(); Uo, F0, Fo, it = vg.mcg_nb(lat.U); t3 = time.time()
        Ua = anneal(random_gauge(lat.U, rng)); t4 = time.time(); Uc = vg.mcg_nb(Ua); t5 = time.time()
        r, gf = measure4(lat, beta, rng, ncopy=1, nsa=0); t6 = time.time()
        print(f"L={L} T={T} b={beta}: sweep {(t1-t0)/5:.2f}s  OR-MCG {t3-t2:.1f}s ({it} it, F {Fo:.5f})  SA {t4-t3:.1f}s "
              f"(F {vg.mcg_functional(Ua):.5f})  OR after SA {t5-t4:.1f}s ({Uc[3]} it, F {Uc[2]:.5f})  measure4(ncopy=1) {t6-t5:.1f}s")
    elif m == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
    elif m == "reanalyze_series2":
        reanalyze_series2()
    elif m == "analyze":
        analyze()
