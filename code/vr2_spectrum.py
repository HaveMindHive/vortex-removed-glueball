#!/usr/bin/env python3
"""Series 4: excited 0++ and 2++ levels on the vortex-removed ensembles, gauge copies at beta 2.4, and beta 2.35.

Imports vr_glueball.py (heat bath, direct maximal center gauge, smearing, Wilson loops, jackknife, plateau rule) and
gauge_copies.py (random gauge copies, best-of-N, string-tension estimators); neither file is modified. New here:
  - plane-resolved basis-A loop operators: A1++ (0++) = sum over the three planes (asserted equal to vr_glueball.loop_ops),
    E++ (2++) = (P_xy - P_xz)/sqrt2 and (P_xy + P_xz - 2 P_yz)/sqrt6, both components pooled in one correlator matrix
  - GEVP levels 0 and 1 (ground and first excited) with the series 2 normalization and (t0, t) = (0, 1)
  - optional paired best-of-N gauge copies (gauge_copies.gauge_fix, no annealing)
Modes: selftest | time L beta ncopy | run L beta seed deadline_unix ncopy | analyze
  run      -> runs_vr2/run_L{L}_b{beta}_s{seed}_c{ncopy}.npz (+ .ckpt); stops at the wall-clock deadline
  analyze  -> runs_vr2/analyze.log, runs_vr2/summary.csv; also reads the series 1 beta 2.4 streams from runs_series1/
"""
import sys, os, glob, importlib.util, pickle, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_vr2")
S1 = os.path.join(HERE, "runs_series1")
os.makedirs(OUT, exist_ok=True)
_spec = importlib.util.spec_from_file_location("gauge_copies", os.path.join(HERE, "gauge_copies.py"))
gc = importlib.util.module_from_spec(_spec); sys.modules["gauge_copies"] = gc; _spec.loader.exec_module(gc)
vg = gc.vg   # the vr_glueball module registered by gauge_copies (one module, so checkpoints pickle vg.Lattice)
mul, dag, sh = vg.mul, vg.dag, vg.sh

# beta 2.35 is new: a sqrt(sigma) interpolated geometrically between the 2.3 and 2.4 table values (smearing scale only)
vg.A_SQS_LIT[2.35] = round(float(np.sqrt(vg.A_SQS_LIT[2.3] * vg.A_SQS_LIT[2.4])), 3)      # 0.313
vg.QLEV[2.35] = tuple(vg.nsmear(2.35)[i] for i in (0, 1, 3))                               # (0, 3, 11), as 2.3 / 2.4
PLANES = ((1, 2), (1, 3), (2, 3))
NSKIP = 2


def loop_planes(F, L, shp=vg.ASHAPES):
    """(nshape, 3 planes, T): zero-momentum loop sums per plane (rectangles: both orientations in the plane)."""
    T = F.shape[2]; out = np.zeros((len(shp), 3, T))
    lines = {(d, n): vg.line(F, d, n) for d in (1, 2, 3) for n in range(1, max(max(s) for s in shp) + 1)}
    for k, (r1, r2) in enumerate(shp):
        for p, (a, b) in enumerate(PLANES):
            for (i, j) in ((a, b), (b, a)):
                if r1 == r2 and (i, j) != (a, b):
                    continue
                W = mul(mul(lines[i, r1], sh(lines[j, r2], i, r1)), dag(mul(lines[j, r2], sh(lines[i, r1], j, r2))))
                out[k, p] += W[0].sum((1, 2, 3))
    return out


def a1(P):  # (..., 3, T) -> (..., T)
    return P.sum(-2)


def e_comp(P):  # (..., 3, T) -> (2, ..., T)
    return np.stack([(P[..., 0, :] - P[..., 1, :]) / np.sqrt(2), (P[..., 0, :] + P[..., 1, :] - 2 * P[..., 2, :]) / np.sqrt(6)])


def measure(lat, beta, rng, ncopy):
    L = lat.L; lev = vg.QLEV[beta]; wlev = vg.nsmear(beta)[1]
    assert wlev == lev[1]
    if ncopy:
        Uo, Un, gf, _ = gc.gauge_fix(lat.U, rng, ncopy, 0)
        gf = gf[[0, 1, 2]].tolist() + [gf[2:].max()]
    else:
        Uo, F0, Fo, it = vg.mcg_nb(lat.U); Un = None; gf = [F0, it, Fo, Fo]
    ens = [("ORIG", lat.U)]
    for tag, Ug in (("", Uo), ("n", Un)):
        if Ug is not None:
            VR, CP = gc.split(Ug); ens += [("VR" + tag, VR), ("CP" + tag, CP)]
    res, pf = {}, {}
    for name, U in ens:
        cp = name.startswith("CP")
        levels = (0,) if cp else lev
        pl, Us, done, Wl = [], U[1:].copy(), 0, None
        for n in levels:
            Us = vg.ape(Us, n - done); done = n
            F = np.concatenate([U[:1], Us])
            pl.append(loop_planes(F, L))
            if n == wlev or cp:
                Wl = vg.wilson(F, U[0], L, L // 2)
        pl = np.array(pl, np.float32); pf[name] = vg.plaq_field(U)
        res[name] = (a1(pl), pl, Wl, pf[name].mean())
    for tag in ("", "n"):
        if "VR" + tag in pf:
            assert np.allclose(pf["ORIG"], pf["CP" + tag] * pf["VR" + tag], atol=1e-8), "center factorization broken"
    return res, gf


def run(L, beta, seed, deadline, ncopy, ntherm=300, ckevery=10):
    fn = f"{OUT}/run_L{L}_b{beta}_s{seed}_c{ncopy}.npz"; ck = fn[:-4] + ".ckpt"
    if os.path.exists(ck):
        lat, D, gfs, ts, rng = pickle.load(open(ck, "rb")); print(f"resumed at {len(ts)} meas", flush=True)
    else:
        lat = vg.Lattice(L, seed=seed); rng = np.random.default_rng(seed + 10000)
        for _ in range(ntherm): lat.sweep(beta)
        print(f"thermalized {time.ctime()}", flush=True)
        D, gfs, ts = {}, [], []
    while time.time() < deadline:
        for _ in range(NSKIP): lat.sweep(beta)
        assert np.allclose(vg.norm(lat.U.swapaxes(0, 1)), 1, atol=1e-10), "links left SU(2)"
        res, gf = measure(lat, beta, rng, ncopy)
        assert gf[3] >= gf[2] >= gf[0], "gauge fixing broken"
        for e, (O, PL, W, P) in res.items():
            for k, x in (("O", O), ("PL", PL), ("W", W), ("P", P)):
                D.setdefault(f"{e}_{k}", []).append(x)
        gfs.append(gf); ts.append(time.time()); i = len(ts)
        if i % ckevery == 0 or time.time() >= deadline:
            pickle.dump((lat, D, gfs, ts, rng), open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
            np.savez(fn + ".tmp.npz", **{k: np.array(v) for k, v in D.items()}, gf=np.array(gfs), ts=np.array(ts), L=L,
                     beta=beta, ncopy=ncopy, nsmear=np.array(vg.QLEV[beta]), shapes=np.array(vg.ASHAPES))
            os.replace(fn + ".tmp.npz", fn)
            g = np.array(gfs)
            print(f"L={L} b={beta} s={seed} c={ncopy} {i} P={np.mean(D['ORIG_P']):.5f} F_old={g[:,2].mean():.5f} "
                  f"F_new={g[:,3].mean():.5f} dt={ts[-1]-ts[-2] if i>1 else 0:.1f}s", flush=True)
    print("deadline reached", flush=True)


# ---------------- analysis ----------------
def gevp_levels(C, nlev=2, t0=0, t1=1):
    """C (Tm+1, n, n) symmetric. series 2 normalization and pruning; returns (nlev, Tm+1) normalized projected correlators."""
    dn = np.sqrt(np.abs(np.diag(C[0]))) + 1e-300; Cn = C / np.outer(dn, dn)[None]
    w, Q = np.linalg.eigh(Cn[t0]); mk = w > 1e-4 * w.max()
    R = Q[:, mk] / np.sqrt(w[mk])
    lam, Y = np.linalg.eigh(R.T @ Cn[t1] @ R)
    out = []
    for k in range(nlev):
        u = R @ Y[:, -1 - k]; c = np.array([u @ Cn[t] @ u for t in range(C.shape[0])]); out.append(c / c[0])
    return np.array(out)


def blocks(ds, preps, X_of, W_of, P_of, subtract=True):
    """per-block (vector, count); vector = [C(t) (pooled over components), op means, W sums, plaquette sum].
    X_of(d) -> (ncomp, N, nop, Lt)."""
    out = []
    for d, (st, nb, _) in zip(ds, preps):
        X = X_of(d)[:, st:].astype(float); W = W_of(d)[st:]; P = P_of(d)[st:]
        nc, N, nop, Lt = X.shape; Tm = Lt // 2
        for idx in np.array_split(np.arange(N), nb):
            x = X[:, idx]
            C = np.stack([np.einsum("cnit,cnjt->ij", x, np.roll(x, -t, -1)) / Lt for t in range(Tm + 1)])
            m = x.mean(-1).sum(1).mean(0) if subtract else np.zeros(nop)    # sum over configs; E: no vacuum subtraction (vev 0 by symmetry)
            out.append((np.concatenate([C.ravel(), m, W[idx].sum(0).ravel(), [P[idx].sum()]]), len(idx)))
    return out, nop, Tm, W.shape[1:], nc


def fns(nop, Tm, wsh, nc):
    nC = (Tm + 1) * nop * nop

    def corr(v):
        C = v[:nC].reshape(Tm + 1, nop, nop) / nc; m = v[nC:nC + nop]
        C = C - np.outer(m, m)[None]
        return 0.5 * (C + C.swapaxes(1, 2))

    def meff(v, k):
        c = gevp_levels(corr(v), 2)[k]
        with np.errstate(all="ignore"):
            return np.log(c[1:Tm] / c[2:Tm + 1])                  # m(1) .. m(Tm-1)
    wl = lambda v: v[nC + nop:nC + nop + wsh[0] * wsh[1]].reshape(wsh)
    return dict(m0=lambda v: meff(v, 0), m1=lambda v: meff(v, 1), wl=wl,
                e1=lambda v: gc.sig_T(wl(v), 3)[1], sqs=lambda v: np.sqrt(gc.sig_T(wl(v), 3)[1]))


def jk_err(est):
    B = len(est); return np.sqrt((B - 1) / B * np.nansum((est - np.nanmean(est, 0)) ** 2, 0))


def load(pattern, src):
    ds = []
    for fn in sorted(glob.glob(pattern)):
        if fn.endswith(".tmp.npz"):
            continue
        d = dict(np.load(fn)); d["src"] = src; d["name"] = os.path.basename(fn); ds.append(d)
    return ds


def X_a1(e, beta):
    def f(d):
        li, si = vg.basis_index(d, beta, "A", e)
        O = d[f"{e}_O"][:, li][:, :, si]; return O.reshape(O.shape[0], -1, O.shape[-1])[None]
    return f


def X_e(e):
    def f(d):
        E = e_comp(d[f"{e}_PL"].astype(float)); return E.reshape(2, E.shape[1], -1, E.shape[-1])
    return f


def channel(ds, preps, beta, e, ch, pr, tag):
    """returns dict: level -> (quoted, err, per-block est, flag, (m, em)); plus E1 and plaquette per block."""
    X = X_a1(e, beta) if ch == "0++" else X_e(e)
    B, nop, Tm, wsh, nc = blocks(ds, preps, X, lambda d: d[f"{e}_W"], lambda d: d[f"{e}_P"], subtract=(ch == "0++"))
    F = fns(nop, Tm, wsh, nc); out = {}
    for k in ((0, 1) if ch == "0++" else (0,)):
        full, err, est = vg.jkw(F["m0" if k == 0 else "m1"], B)
        v, ev, vest, fl = vg.plateau((full, err), est)
        out[k] = (v, ev, vest, fl, (full, err))
        pr(f"  {tag} {e:5s} {ch} level {k}: m(t) = " + ", ".join(f"{a:.3f}({b:.3f})" for a, b in zip(full[:4], err[:4]))
           + f"  -> quoted {v:.3f}({ev:.3f}) {fl}")
    out["E1"] = vg.jkw(F["e1"], B)
    return out


def ratio(a, b):
    """quoted masses with aligned per-block estimates -> (r, err, est)."""
    r = a[0] / b[0]; est = a[2] / b[2]; return r, jk_err(est), est


def quad(x, y, op):
    if op == "/":
        r = x[0] / y[0]; return r, abs(r) * np.hypot(x[1] / x[0], y[1] / y[0])
    return x[0] - y[0], np.hypot(x[1], y[1])


def analyze():
    lines = []
    def pr(s=""):
        print(s); lines.append(s)
    rows = []
    def row(ds_name, beta, L, e, ch, k, res, N):
        v, ev, _, fl, (m, em) = res
        rows.append([ds_name, beta, L, e, ch, k, N] + [f"{m[i]:.4f}" for i in range(3)] + [f"{em[i]:.4f}" for i in range(3)]
                    + [f"{v:.4f}", f"{ev:.4f}", fl])
    R = {}
    sets = [("D-S1", 2.4, 12, load(f"{S1}/exploratory/run_L12_b2.4_*.npz", "S1x") + load(f"{S1}/run_L12_b2.4_*.npz", "S1"), ("ORIG", "VR", "CP"), ("0++",)),
            ("D-S1", 2.4, 16, load(f"{S1}/run_L16_b2.4_*.npz", "S1"), ("ORIG", "VR", "CP"), ("0++",)),
            ("D-24", 2.4, 12, load(f"{OUT}/run_L12_b2.4_s70*_c*.npz", "S4"), ("ORIG", "VR", "CP"), ("0++", "2++")),
            ("D-copy", 2.4, 12, load(f"{OUT}/run_L12_b2.4_s701_c7.npz", "S4"), ("ORIG", "VR", "VRn", "CP", "CPn"), ("0++",)),
            ("D-235", 2.35, 10, load(f"{OUT}/run_L10_b2.35_*.npz", "S4"), ("ORIG", "VR", "CP"), ("0++", "2++"))]
    for name, beta, L, ds, ens, chs in sets:
        if not ds:
            pr(f"{name} beta {beta} L {L}: no data"); continue
        pr(f"== {name} beta {beta} L {L}: {len(ds)} streams")
        preps = [vg.stream_prep(d, pr) for d in ds]; N = sum(len(d["ORIG_P"]) - p[0] for d, p in zip(ds, preps))
        pr(f"  N = {N}, <P_ORIG> = {np.mean(np.concatenate([d['ORIG_P'] for d in ds])):.5f}")
        if "gf" in ds[0] and ds[0]["gf"].shape[1] == 4:
            g = np.concatenate([d["gf"] for d in ds]); gain = g[:, 3] - g[:, 2]
            pr(f"  MCG functional single {g[:,2].mean():.5f}, best {g[:,3].mean():.5f}, gain {gain.mean():.5f}"
               f"({gain.std()/np.sqrt(len(gain)):.5f}), best>single on {np.mean(gain > 1e-9):.2f}")
            R[name, "gainfrac"] = np.mean(gain > 1e-9)
        for e in ens:
            for ch in chs:
                if e.startswith("CP") and ch == "2++":
                    continue
                if e.startswith("CP") and ch == "0++":
                    # CP: only the string tension is used (C0); glueball ops for CP not stored in series 4 runs
                    B, nop, Tm, wsh, nc = blocks(ds, preps, lambda d: d[f"{e}_O"][:, :1, :1].reshape(len(d[f"{e}_O"]), 1, -1)[None],
                                                  lambda d: d[f"{e}_W"], lambda d: d[f"{e}_P"])
                    R[name, L, e, "E1"] = vg.jkw(fns(nop, Tm, wsh, nc)["e1"], B); continue
                out = channel(ds, preps, beta, e, ch, pr, f"{name} L{L}")
                R[name, L, e, ch] = out; R[name, L, e, "E1"] = out["E1"]
                for k in out:
                    if k != "E1":
                        row(name, beta, L, e, ch, k, out[k], N)
        o = R[name, L, "ORIG", "E1"]
        R[name, L, "sqs"] = (np.sqrt(o[0]), jk_err(np.sqrt(o[2])), np.sqrt(o[2]))
        pr(f"  a sqrt(sigma) ORIG (E1) = {R[name, L, 'sqs'][0]:.4f}({R[name, L, 'sqs'][1]:.4f})")
        for e in ens:
            if e != "ORIG":
                r = o[2]; x = R[name, L, e, "E1"]; est = x[2] / r
                pr(f"  E1 ratio {e}/ORIG = {x[0]/o[0]:.3f}({jk_err(est):.3f})")
                R[name, L, e, "r"] = (x[0] / o[0], jk_err(est), est)
        for e in ens:
            if e.startswith("VR"):
                f = ratio(R[name, L, e, "0++"][0], R[name, L, "ORIG", "0++"][0]); R[name, L, e, "f"] = f
                ms = ratio(R[name, L, e, "0++"][0], R[name, L, "sqs"])
                pr(f"  fraction m_{e}/m_ORIG (0++ ground) = {f[0]:.3f}({f[1]:.3f});  m_{e}/sqrt(sigma) = {ms[0]:.3f}({ms[1]:.3f})")
                for ch, k in (("0++", 1), ("2++", 0)):
                    if (name, L, e, ch) in R:
                        x = ratio(R[name, L, e, ch][k], R[name, L, "ORIG", ch][k])
                        pr(f"  m_{e}/m_ORIG {ch} level {k} = {x[0]:.3f}({x[1]:.3f}); minus ground fraction "
                           f"{x[0]-f[0]:+.3f}({jk_err(x[2]-f[2]):.3f})")
                        R[name, L, e, ch, k, "rx"] = (x[0] - f[0], jk_err(x[2] - f[2]))
                        y = ratio(R[name, L, e, ch][k], R[name, L, e, "0++"][0])
                        pr(f"  m_{e}({ch},{k}) / m_{e}(0++,0) = {y[0]:.3f}({y[1]:.3f}); ORIG same ratio "
                           f"{ratio(R[name, L, 'ORIG', ch][k], R[name, L, 'ORIG', '0++'][0])[0]:.3f}")
        pr(f"  free two-gluon level 4 sin(pi/L) = {4*np.sin(np.pi/L):.3f}")
    pr("\n== decisions (pre-registered rules)")
    def sig(x, ref, e):
        return abs(x - ref) / e
    # R1-P1a / R4 excited 0++ (D-S1)
    if ("D-S1", 16, "VR", "0++") in R and ("D-S1", 12, "VR", "0++") in R:
        a, b = R["D-S1", 16, "VR", "0++"][1], R["D-S1", 12, "VR", "0++"][1]
        rs, ers = quad(a, b, "/")
        n1, nb = sig(rs, 1, ers), sig(rs, 0.754, ers)
        v = "KILLED" if (n1 < 2 and nb > 2) else ("SUPPORTED (box level)" if rs < 1 and n1 > 2 else "INCONCLUSIVE")
        if "NOISY" in (a[3], b[3]): v = "NOT TESTED (NOISY)"
        pr(f"R1-P1a: excited VR 0++ 12^4 {b[0]:.3f}({b[1]:.3f}) {b[3]}, 16^4 {a[0]:.3f}({a[1]:.3f}) {a[3]}; rho* = {rs:.3f}({ers:.3f})"
           f" ({n1:.1f} sigma from 1, {nb:.1f} from 0.754) -> {v}")
        ao, bo = R["D-S1", 16, "ORIG", "0++"][1], R["D-S1", 12, "ORIG", "0++"][1]
        ro, ero = quad(ao, bo, "/")
        pr(f"  (report) excited ORIG 0++ 12^4 {bo[0]:.3f}({bo[1]:.3f}) {bo[3]}, 16^4 {ao[0]:.3f}({ao[1]:.3f}) {ao[3]}; ratio {ro:.3f}({ero:.3f})")
        for L in (12, 16):
            d, e = R["D-S1", L, "VR", "0++", 1, "rx"]
            pr(f"R4 excited 0++ L={L}: (m_VR/m_ORIG)_excited - (m_VR/m_ORIG)_ground = {d:+.3f}({e:.3f}), {abs(d)/e:.1f} sigma")
    for name, L in (("D-24", 12), ("D-235", 10)):
        if (name, L, "VR", "2++") in R:
            m2, m0 = R[name, L, "VR", "2++"][0], R[name, L, "VR", "0++"][0]
            pr(f"{name} L={L}: VR 2++ {m2[0]:.3f}({m2[1]:.3f}) {m2[3]}; VR 0++ {m0[0]:.3f}({m0[1]:.3f}); ORIG 2++ "
               f"{R[name, L, 'ORIG', '2++'][0][0]:.3f}({R[name, L, 'ORIG', '2++'][0][1]:.3f}) {R[name, L, 'ORIG', '2++'][0][3]}")
            d = m2[2] - m0[2]; dd, ed = m2[0] - m0[0], jk_err(d); th = 4 * np.sin(np.pi / L)
            kill = dd > 2 * ed and m2[0] < th - 2 * m2[1]
            pr(f"  R1-P1b: m2 - m0 = {dd:+.3f}({ed:.3f}); m2 vs box level {th:.3f}: {(m2[0]-th)/m2[1]:+.1f} sigma -> "
               f"{'NOT TESTED (NOISY)' if m2[3]=='NOISY' else ('KILLED' if kill else 'not killed')}")
            d, e = R[name, L, "VR", "2++", 0, "rx"]
            pr(f"  R4 2++: (m_VR/m_ORIG)_2++ - (m_VR/m_ORIG)_0++ = {d:+.3f}({e:.3f}), {abs(d)/e:.1f} sigma")
    # G24
    if ("D-copy", 12, "VRn", "0++") in R:
        a, b = R["D-copy", 12, "VRn", "0++"][0], R["D-copy", 12, "VR", "0++"][0]
        dq = a[0] - b[0]; eq = jk_err(a[2] - b[2])
        m1a, m1b = R["D-copy", 12, "VRn", "0++"][0][4], R["D-copy", 12, "VR", "0++"][0][4]
        pr(f"G24: VR quoted single {b[0]:.3f}({b[1]:.3f}) {b[3]}, best-of-8 {a[0]:.3f}({a[1]:.3f}) {a[3]}; paired diff {dq:+.3f}({eq:.3f})"
           f" = {abs(dq)/eq:.1f} sigma; m(1) single {m1b[0][0]:.3f}, best {m1a[0][0]:.3f}")
        for e in ("VR", "VRn", "CP", "CPn"):
            x = R["D-copy", 12, e, "r"]; pr(f"  E1 ratio {e}/ORIG = {x[0]:.3f}({x[1]:.3f})")
        for a_, b_ in (("VRn", "VR"), ("CPn", "CP")):
            x, y = R["D-copy", 12, a_, "r"], R["D-copy", 12, b_, "r"]
            pr(f"  paired E1 ratio {a_}-{b_} = {x[0]-y[0]:+.3f}({jk_err(x[2]-y[2]):.3f})")
        fa, fb = R["D-copy", 12, "VRn", "f"], R["D-copy", 12, "VR", "f"]
        pr(f"  fraction single {fb[0]:.3f}({fb[1]:.3f}), best {fa[0]:.3f}({fa[1]:.3f}), paired {fa[0]-fb[0]:+.3f}({jk_err(fa[2]-fb[2]):.3f})")
        R["G24"] = "NOT TESTED" if R.get(("D-copy", "gainfrac"), 0) < 0.8 else ("COPY-STABLE" if abs(dq) < 2 * eq else "COPY-SENSITIVE")
        pr(f"  -> G24 {R['G24']}")
    # R2, R3
    if ("D-235", 10, "VR", "0++") in R:
        rv, rc = R["D-235", 10, "VR", "r"], R["D-235", 10, "CP", "r"]
        mo, mv = R["D-235", 10, "ORIG", "0++"][0], R["D-235", 10, "VR", "0++"][0]
        c0 = rv[0] < 0.15 and 0.6 <= rc[0] <= 1.4 and mo[3] in ("plateau", "late") and mv[3] in ("plateau", "late")
        pr(f"C0-235: VR E1 ratio {rv[0]:.3f}({rv[1]:.3f}), CP {rc[0]:.3f}({rc[1]:.3f}), ORIG {mo[0]:.3f}({mo[1]:.3f}) {mo[3]}, "
           f"VR {mv[0]:.3f}({mv[1]:.3f}) {mv[3]} -> {'PASS' if c0 else 'FAIL'}")
        f35, f24 = R["D-235", 10, "VR", "f"], R["D-S1", 12, "VR", "f"]
        df, edf = quad(f35, f24, "-"); f16 = R["D-S1", 16, "VR", "f"]
        pr(f"R3: f(2.35, 10^4) = {f35[0]:.3f}({f35[1]:.3f}); f(2.4, 12^4) = {f24[0]:.3f}({f24[1]:.3f}); diff {df:+.3f}({edf:.3f}) = "
           f"{abs(df)/edf:.1f} sigma -> {('HOLDS' if abs(df) < 2*edf else 'KILLED') if c0 else 'NOT TESTED'}"
           f"; vs 16^4 {f16[0]:.3f}({f16[1]:.3f}): {abs(f35[0]-f16[0])/np.hypot(f35[1], f16[1]):.1f} sigma")
        s, se = quad(R["D-235", 10, "sqs"], R["D-S1", 12, "sqs"], "/")
        ra, rae = quad(mv, R["D-S1", 12, "VR", "0++"][0], "/")
        n1, ns = abs(ra - 1) / rae, abs(ra - s) / np.hypot(rae, se)
        v = "SUPPORTED" if (n1 < 2 and ns > 2) else ("KILLED" if (ns < 2 and n1 > 2) else "INCONCLUSIVE")
        pr(f"R2: s = a sqrt(sigma) ratio {s:.3f}({se:.3f}); rho_a = {ra:.3f}({rae:.3f}); {n1:.1f} sigma from 1, {ns:.1f} from s -> "
           f"{v if c0 else 'NOT TESTED'}")
        ms35 = quad(mv, R["D-235", 10, "sqs"], "/"); ms24 = quad(R["D-S1", 12, "VR", "0++"][0], R["D-S1", 12, "sqs"], "/")
        pr(f"  m_VR/sqrt(sigma): 2.35 {ms35[0]:.3f}({ms35[1]:.3f}), 2.4 12^4 {ms24[0]:.3f}({ms24[1]:.3f})")
        R["R3"] = ("HOLDS" if abs(df) < 2 * edf else "KILLED") if c0 else "NOT TESTED"
    if "R3" in R and "G24" in R:
        comb = "ROBUST (in this window)" if (R["R3"] == "HOLDS" and R["G24"] == "COPY-STABLE") else (
            "NOT ROBUST" if (R["R3"] == "KILLED" or R["G24"] == "COPY-SENSITIVE") else "OPEN")
        pr(f"Combined: fraction {comb}")
    open(f"{OUT}/analyze.log", "w").write("\n".join(lines) + "\n")
    import csv
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "beta", "L", "ensemble", "channel", "gevp_level", "N", "m1", "m2", "m3", "e_m1", "e_m2", "e_m3",
                    "quoted", "e_quoted", "flag"])
        w.writerows(rows)


def selftest():
    rng = np.random.default_rng(7)
    lat = vg.Lattice(6, seed=3)
    for _ in range(20): lat.sweep(2.4)
    F = np.concatenate([lat.U[:1], vg.ape(lat.U[1:], 2)])
    PL = loop_planes(F, 6)
    full = vg.loop_ops(F, 6); si = [vg.shapes(6).index(s) for s in vg.ASHAPES if s in vg.shapes(6)]
    shp6 = [s for s in vg.ASHAPES if s in vg.shapes(6)]
    assert np.allclose(a1(loop_planes(F, 6, shp6)), full[si]), "A1 != vr_glueball loop_ops"
    # E components are orthogonal to A1 and transform among themselves: under the x<->y swap planes (xy,xz,yz)->(xy,yz,xz)
    E = e_comp(PL); Es = e_comp(PL[:, [0, 2, 1]])
    nE = (E ** 2).sum(0); assert np.allclose(nE, (Es ** 2).sum(0)), "E norm not invariant"
    # GEVP level 0 equals vr_glueball's gevp_corr on a random positive correlator
    A = rng.normal(size=(5, 5)); masses = np.array([0.5, 0.9, 1.3, 1.7, 2.1])
    C = np.stack([A @ np.diag(np.exp(-masses * t)) @ A.T for t in range(5)])
    g = gevp_levels(C, 2)
    assert np.allclose(np.log(g[0][1] / g[0][2]), 0.5, atol=1e-8) and np.allclose(np.log(g[1][1] / g[1][2]), 0.9, atol=1e-8)
    nC = 5 * 25; v = np.concatenate([C.ravel(), np.zeros(5), np.ones(4), [0]])
    vgg = vg.make_fns(5, 4, (2, 2))["gevp_corr"](v, 0, 1)
    assert np.allclose(vgg, g[0]), "GEVP level 0 != vr_glueball"
    # measure on a tiny lattice: single copy and best-of-2, center factorization asserted inside
    for nc in (0, 2):
        r, gf = measure(lat, 2.4, rng, nc)
        assert r["ORIG"][1].shape == (3, len(vg.ASHAPES), 3, 6) and gf[3] >= gf[2]
    assert "VRn" in r and r["VRn"][2].shape == (3, 3)
    # single-copy VR equals vr_glueball.measure VR Wilson loops (same gauge fixing, same level)
    r3, _ = vg.measure(lat, 2.4)
    r0, _ = measure(lat, 2.4, rng, 0)
    assert np.allclose(r3["VR"][1], r0["VR"][2]) and np.allclose(r3["ORIG"][1], r0["ORIG"][2])
    print("QLEV 2.35 =", vg.QLEV[2.35], "a_sqs 2.35 =", vg.A_SQS_LIT[2.35])
    print("selftest OK")


def timeit(L, beta, ncopy):
    lat = vg.Lattice(L, seed=99); rng = np.random.default_rng(1)
    for _ in range(5): lat.sweep(beta)
    t = time.time(); [lat.sweep(beta) for _ in range(NSKIP)]; t1 = time.time()
    measure(lat, beta, rng, ncopy); t2 = time.time()
    print(f"L={L} b={beta} c={ncopy}: {NSKIP} sweeps {t1-t:.1f}s, measure {t2-t1:.1f}s (unthermalized config)")


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "selftest":
        selftest()
    elif m == "time":
        timeit(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    elif m == "run":
        run(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))
    elif m == "analyze":
        analyze()
