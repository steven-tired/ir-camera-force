"""Global-consensus + bundle-adjustment refinement of the Lepton->D435i thermal->color extrinsic.

Two-stage robust estimator on the existing both-detected checkerboard pairs (no recapture):
  Stage 1  per-pair candidate rig transforms from IPPE planar-pose branches x thermal orderings.
  Stage 2  deterministic max-inlier global consensus (rotation+translation) -> frozen assignments.
  Stage 3  bundle adjustment (scipy) minimizing color+thermal reprojection -> tight R,T.
  Stage 4  honest metric: symmetric epipolar distance (planar-pose-branch independent) + LOO.

Thermal intrinsics are FIXED to FLIR's published Lepton 3.1R Brown-Conrady params (do not re-fit);
color is the D435i factory pinhole. See scratch_lepton/PLAN_refine_extrinsic.md.

Run:
  cd $WORKSPACE_ROOT
  env -u PYTHONPATH .venv-lerobot/bin/python scripts/refine_extrinsic.py \
    --build <run>/stream/build [--out <dir>] [--stage 2|3|4]
"""
import argparse
import glob
import os

import cv2
import numpy as np

SERIAL = "233522078685"
PATTERN = (4, 3)
SB_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
OBJ = np.array([[c * 0.03, r * 0.03, 0] for r in range(3) for c in range(4)], np.float64)

# FLIR official Lepton 3.1R Brown-Conrady (dewarp application note)
KT = np.array([[104.654, 0, 79.123], [0, 104.483, 55.689], [0, 0, 1]], float)
DT = np.array([-0.39758, 0.18069, 0.00463, 0.00420, -0.03381], float).reshape(-1, 1)

# --- frozen numeric config v1 ---
# rot_gate does the discrete flip/branch disambiguation (a wrong 180deg flip lands ~180deg away).
# trans_gate only rejects gross outliers among rotation-compatible candidates, so it is loose
# (per-pair PnP translation on this small near-frontal board naturally scatters ~1.5 cm).
CFG = dict(rot_gate_deg=15.0, trans_gate_mm=25.0, dedup_rot_deg=1.0, dedup_trans_mm=2.0,
           phys_cap_mm=300.0, color_reproj_px=3.0, thermal_reproj_px=3.0, z_eps=1e-4,
           max_rejected_frac=0.15, min_retained=20, mode_rot_deg=10.0, mode_trans_mm=20.0)


def sb(gray):
    ok, c = cv2.findChessboardCornersSB(gray, PATTERN, flags=SB_FLAGS)
    if ok and c is not None and len(c) == 12:
        return c.reshape(-1, 2).astype(np.float64)
    return None


def ang_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1, 1)))


def color_intrinsics():
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
    prof = pipe.start(cfg)
    ci = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pipe.stop()
    K = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1]], float)
    return K, np.zeros((5, 1))


def ippe_poses(pts, K, D, reproj_gate):
    """Return list of (R, t) planar-pose branches that are front-facing + within reproj gate."""
    out = []
    try:
        n, rvs, tvs, _ = cv2.solvePnPGeneric(OBJ, pts, K, D, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        n = 0
    for k in range(n):
        R, _ = cv2.Rodrigues(rvs[k])
        Xc = (R @ OBJ.T + tvs[k]).T
        if np.any(Xc[:, 2] <= CFG["z_eps"]):
            continue
        pr, _ = cv2.projectPoints(OBJ, rvs[k], tvs[k], K, D)
        e = np.sqrt(((pr.reshape(-1, 2) - pts) ** 2).sum(1)).mean()
        if e <= reproj_gate:
            out.append((R, tvs[k].reshape(3, 1), float(e)))
    if not out:  # fallback: ITERATIVE single solution (keep even if slightly over gate)
        ok, rv, tv = cv2.solvePnP(OBJ, pts, K, D)
        if ok:
            R, _ = cv2.Rodrigues(rv)
            out.append((R, tv.reshape(3, 1), 1e9))
    return out


def reverse_grid(pts):
    # flattened row-major 4x3 reverse == (r,c)->(H-1-r,W-1-c)
    return pts[::-1].copy()


def gen_candidates(cc, tc, Kc, Dc):
    """Up to 8 rig candidates for one pair: (R_g, T_g, order_id, Rc, tc_pose, Rt, tt_pose, tpts)."""
    color_poses = ippe_poses(cc, Kc, Dc, CFG["color_reproj_px"])
    cands = []
    for order_id, tpts in enumerate((tc, reverse_grid(tc))):
        tposes = ippe_poses(tpts, KT, DT, CFG["thermal_reproj_px"])
        for (Rc, tcp, _) in color_poses:
            for (Rt, ttp, _) in tposes:
                G = Rc @ Rt.T
                Tg = tcp - G @ ttp
                if np.linalg.norm(Tg) * 1000.0 > CFG["phys_cap_mm"]:
                    continue
                cands.append(dict(R=G, T=Tg, order=order_id, tpts=tpts,
                                  Rc=Rc, tcp=tcp, Rt=Rt, ttp=ttp))
    # dedup
    keep = []
    for c in cands:
        dup = False
        for k in keep:
            if ang_deg(k["R"].T @ c["R"]) < CFG["dedup_rot_deg"] and \
               np.linalg.norm(k["T"] - c["T"]) * 1000.0 < CFG["dedup_trans_mm"]:
                dup = True
                break
        if not dup:
            keep.append(c)
    return keep


def consensus(pair_cands):
    """Deterministic max-inlier consensus. Returns (R, T, inlier_idx, chosen[per inlier])."""
    rg = np.radians(CFG["rot_gate_deg"])
    tg = CFG["trans_gate_mm"] / 1000.0
    seeds = [(c["R"], c["T"]) for cands in pair_cands for c in cands]

    def score(Rs, Ts):
        inl, chosen = [], {}
        for i, cands in enumerate(pair_cands):
            best = None
            for c in cands:
                if np.radians(ang_deg(Rs.T @ c["R"])) <= rg:
                    d = np.linalg.norm(Ts - c["T"])
                    if best is None or d < best[0]:
                        best = (d, c)
            if best is not None and best[0] <= tg:
                inl.append(i)
                chosen[i] = best[1]
        return inl, chosen

    ranked = []
    for (Rs, Ts) in seeds:
        inl, chosen = score(Rs, Ts)
        if inl:
            rr = np.median([ang_deg(Rs.T @ chosen[i]["R"]) for i in inl])
            ranked.append((len(inl), -rr, Rs, Ts, inl, chosen))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, Rs, Ts, inl, chosen = ranked[0]

    # iterate: robust refine + reassign to fixed point
    R, T = Rs, Ts
    for _ in range(10):
        Rmats = [chosen[i]["R"] for i in inl]
        R = chordal_mean(Rmats)
        T = np.median(np.hstack([chosen[i]["T"] for i in inl]), axis=1).reshape(3, 1)
        inl2, chosen2 = score(R, T)
        if set(inl2) == set(inl):
            inl, chosen = inl2, chosen2
            break
        inl, chosen = inl2, chosen2
    # runner-up distinct-mode margin
    margin = distinct_margin(ranked, R, T, len(inl))
    return R, T, inl, chosen, margin


def chordal_mean(Rmats):
    M = np.zeros((3, 3))
    for R in Rmats:
        M += R
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def distinct_margin(ranked, R, T, win_inl):
    for (n, _, Rs, Ts, inl, ch) in ranked:
        if ang_deg(R.T @ Rs) > CFG["mode_rot_deg"] or \
           np.linalg.norm(T - Ts) * 1000.0 > CFG["mode_trans_mm"]:
            return win_inl - n  # inlier-count margin over best distinct runner-up
    return win_inl  # no distinct runner-up


def bundle_adjust(pairs, Kc, Dc, R0, T0, inl, chosen):
    """Refine rig R,T + per-pair color board poses by minimizing color+thermal reprojection."""
    from scipy.optimize import least_squares
    idx = list(inl)
    # init params: rig rvec(3), T(3), then per-pair rvec(3)+tvec(3) from chosen color pose
    rvec0, _ = cv2.Rodrigues(R0)
    p0 = [rvec0.ravel(), T0.ravel()]
    cc_list, tpts_list = [], []
    for i in idx:
        rv, _ = cv2.Rodrigues(chosen[i]["Rc"])
        p0.append(rv.ravel())
        p0.append(chosen[i]["tcp"].ravel())
        cc_list.append(pairs[i][1])            # color corners
        tpts_list.append(chosen[i]["tpts"])    # chosen thermal ordering
    p0 = np.concatenate(p0)

    def residuals(p):
        rvec = p[0:3]
        T = p[3:6].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        res = []
        for j in range(len(idx)):
            rv = p[6 + 6 * j: 9 + 6 * j]
            tv = p[9 + 6 * j: 12 + 6 * j].reshape(3, 1)
            Rc, _ = cv2.Rodrigues(rv)
            # color reprojection
            prc, _ = cv2.projectPoints(OBJ, rv, tv, Kc, Dc)
            res.append((prc.reshape(-1, 2) - cc_list[j]).ravel())
            # thermal: board->color->thermal, project with FLIR K,D
            Xc = (Rc @ OBJ.T + tv)                      # 3xN in color frame
            Xt = (R.T @ (Xc - T)).T                     # Nx3 in thermal frame
            prt, _ = cv2.projectPoints(Xt.astype(np.float64), np.zeros(3), np.zeros(3), KT, DT)
            res.append((prt.reshape(-1, 2) - tpts_list[j]).ravel())
        return np.concatenate(res)

    r0 = residuals(p0)
    sol = least_squares(residuals, p0, loss="soft_l1",
                        f_scale=max(1.0, np.median(np.abs(r0))), max_nfev=200)
    rf = sol.x
    R, _ = cv2.Rodrigues(rf[0:3])
    T = rf[3:6].reshape(3, 1)
    # per-camera RMS before/after
    def rms(res):
        r = res.reshape(-1, 2)
        return np.sqrt((r ** 2).sum(1)).mean()
    return R, T, rms(r0), rms(sol.fun)


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], float)


def epipolar_px(pairs, Kc, Dc, R, T):
    """Branch-free symmetric epipolar (Sampson) distance in color-pixel-equivalent, per pair.
    Uses only point correspondences + E=[T]xR; independent of the ambiguous planar board pose.
    Marginalizes SB's meaningless 180deg thermal label by taking the lower-error ordering."""
    E = skew(T.ravel()) @ R  # x_c^T E x_t = 0 for X_c = R X_t + T (normalized coords)
    fc = (Kc[0, 0] + Kc[1, 1]) / 2.0
    out = []
    for (_, cc, tc) in pairs:
        uc = cv2.undistortPoints(cc.reshape(-1, 1, 2), Kc, Dc).reshape(-1, 2)
        best = None
        for tpts in (tc, tc[::-1].copy()):
            ut = cv2.undistortPoints(tpts.reshape(-1, 1, 2), KT, DT).reshape(-1, 2)
            xc = np.hstack([uc, np.ones((12, 1))])
            xt = np.hstack([ut, np.ones((12, 1))])
            Ext = (E @ xt.T).T           # 12x3
            Etxc = (E.T @ xc.T).T
            num = np.einsum('ij,ij->i', xc, Ext) ** 2
            den = Ext[:, 0]**2 + Ext[:, 1]**2 + Etxc[:, 0]**2 + Etxc[:, 1]**2
            samp = np.sqrt(num / np.maximum(den, 1e-12)) * fc   # color-px equiv
            m = samp.mean()
            if best is None or m < best:
                best = m
                bestmax = samp.max()
        out.append((best, bestmax))
    return np.array(out)


def load_pairs(build):
    CS, TS = f"{build}/images", f"{build}/thermal_images"
    pairs = []
    for cp in sorted(glob.glob(f"{CS}/color_image_*.png"),
                     key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))))):
        i = int("".join(filter(str.isdigit, os.path.basename(cp))))
        cg = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
        tg = cv2.imread(f"{TS}/thermal_grayimage_{i}.png", cv2.IMREAD_GRAYSCALE)
        if cg is None or tg is None:
            continue
        cc, tc = sb(cg), sb(tg)
        if cc is not None and tc is not None:
            pairs.append((i, cc, tc))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = load_pairs(args.build)
    print(f"loaded {len(pairs)} both-detected pairs")
    Kc, Dc = color_intrinsics()
    print(f"color K: fx={Kc[0,0]:.1f} cx={Kc[0,2]:.1f} cy={Kc[1,2]:.1f}")

    pair_cands = [gen_candidates(cc, tc, Kc, Dc) for (_, cc, tc) in pairs]
    ncand = [len(c) for c in pair_cands]
    print(f"Stage1: candidates/pair min={min(ncand)} med={int(np.median(ncand))} max={max(ncand)}")

    R, T, inl, chosen, margin = consensus(pair_cands)
    rej = 1 - len(inl) / len(pairs)
    print(f"Stage2: inliers {len(inl)}/{len(pairs)} (rejected {rej:.2f})  "
          f"|T|={np.linalg.norm(T)*100:.2f}cm  rigRot={ang_deg(R):.1f}deg  "
          f"runnerup_margin={margin} pairs")
    # per-pair consistency of chosen candidates
    Ts = np.hstack([chosen[i]["T"] for i in inl])
    Rs = [ang_deg(R.T @ chosen[i]["R"]) for i in inl]
    print(f"  chosen per-pair T spread std(cm)={np.round(Ts.std(1)*100,2)}  "
          f"rot dispersion max={max(Rs):.1f}deg")
    if rej > CFG["max_rejected_frac"]:
        print(f"  WARNING rejection {rej:.2f} > {CFG['max_rejected_frac']}")
    if len(inl) < CFG["min_retained"]:
        print(f"  WARNING retained {len(inl)} < {CFG['min_retained']}")

    if args.stage < 3:
        print("Stage2 done.")
        return

    Rb, Tb, rms0, rms1 = bundle_adjust(pairs, Kc, Dc, R, T, inl, chosen)
    print(f"Stage3 (BA): reproj RMS {rms0:.2f}px -> {rms1:.2f}px  "
          f"|T|={np.linalg.norm(Tb)*100:.2f}cm  rigRot={ang_deg(Rb):.1f}deg")
    print(f"  T(cm)={np.round(Tb.ravel()*100,2)}")

    if args.stage < 4:
        print("Stage3 done.")
        return

    # Stage 4a: branch-free epipolar metric over ALL pairs (honest, not fooled by planar-pose branch)
    ep = epipolar_px(pairs, Kc, Dc, Rb, Tb)
    print(f"Stage4 epipolar (color-px equiv, branch-free, all {len(pairs)} pairs): "
          f"median={np.median(ep[:,0]):.2f}  p95={np.percentile(ep[:,0],95):.2f}  "
          f"max={ep[:,0].max():.2f}")
    inl_ep = ep[list(inl), 0]
    print(f"  inliers-only: median={np.median(inl_ep):.2f} p95={np.percentile(inl_ep,95):.2f}")

    # Stage 4b: leave-one-out stability (re-run consensus+BA leaving each inlier out)
    Rs, Ts = [], []
    for drop in list(inl):
        sub = [c for k, c in enumerate(pair_cands) if k != drop]
        Rc2, Tc2, inl2, ch2, _ = consensus(sub)
        # map sub indices back to original for BA
        origmap = [k for k in range(len(pair_cands)) if k != drop]
        inl2o = [origmap[i] for i in inl2]
        ch2o = {origmap[i]: ch2[i] for i in inl2}
        try:
            Rf, Tf, _, _ = bundle_adjust(pairs, Kc, Dc, Rc2, Tc2, inl2o, ch2o)
            Rs.append(ang_deg(Rb.T @ Rf)); Ts.append(np.linalg.norm(Tb - Tf) * 100)
        except Exception:
            pass
    Rs, Ts = np.array(Rs), np.array(Ts)
    print(f"Stage4 LOO stability ({len(Rs)} runs): dR median={np.median(Rs):.2f}deg max={Rs.max():.2f}  "
          f"d|T| median={np.median(Ts):.2f}cm max={Ts.max():.2f}")

    # save
    # build is <run>/stream/build -> run dir is two levels up
    run_dir = os.path.dirname(os.path.dirname(args.build.rstrip('/')))
    outdir = args.out or os.path.join(run_dir, "calibration", "FINAL_flir_brown")
    os.makedirs(outdir, exist_ok=True)
    fe = cv2.FileStorage(os.path.join(outdir, "extrinsic_refined.xml"), cv2.FILE_STORAGE_WRITE)
    fe.write("R", Rb); fe.write("T", Tb); fe.write("direction", "thermal_to_color")
    fe.write("unit", "meter"); fe.write("color_K", Kc)
    fe.write("thermal_K", KT); fe.write("thermal_D", DT)
    fe.write("method", "global-consensus + bundle-adjustment (refine_extrinsic.py)")
    fe.write("rigRot_deg", float(ang_deg(Rb))); fe.write("T_norm_cm", float(np.linalg.norm(Tb) * 100))
    fe.write("ba_reproj_rms_px", float(rms1))
    fe.write("epipolar_median_px", float(np.median(ep[:, 0])))
    fe.write("epipolar_p95_px", float(np.percentile(ep[:, 0], 95)))
    fe.write("loo_dT_median_cm", float(np.median(Ts)) if len(Ts) else -1.0)
    fe.write("inlier_count", int(len(inl)))
    fe.write("note", "PROVISIONAL - not actuator-qualified. Thermal=FLIR Brown raw; color=D435i pinhole.")
    fe.release()
    print(f"saved {outdir}/extrinsic_refined.xml")
    print("Stage4 done.")


if __name__ == "__main__":
    main()
