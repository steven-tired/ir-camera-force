#!/usr/bin/env bash
# Compute Lepton intrinsics + Lepton<->RealSense extrinsics from ALREADY-captured
# color/thermal pairs, using the SB-detector build of the frozen tools.
#
# This is a pragmatic compute path (NOT the sealed immutable runbook). It:
#   1. picks the captured pairs where findChessboardCornersSB detects a full 4x3
#      board in BOTH the color and thermal image,
#   2. stages 24 as the fit set + 12 as a disjoint held-out set,
#   3. runs camera_calibration (thermal intrinsics) -> extrinsic (thermal<->RGB)
#      -> heldout_verify (reprojection gate),
#   4. prints calibration.xml, extrinsic.xml, and the held-out max error.
#
# Requires: the D435i (233522078685) connected (extrinsic + heldout read its live
# color intrinsics), and the verifier worktree built. Re-runnable.
#
# Usage:
#   scripts/compute_calib_from_captures.sh [CAPTURES_DIR] [VERIFIER_WT]
# Defaults: newest run's stream/build captures, newest verifier worktree.
set -uo pipefail

ROOT=/home/zhuokai/hand-teleop/thermal-project-calibration-runs
PY=/home/zhuokai/hand-teleop/.venv-lerobot/bin/python
RS_SERIAL=233522078685
CAP="${1:-$(ls -dt "$ROOT"/worktrees/*/stream/build 2>/dev/null | head -1)}"
WT="${2:-$(ls -dt "$ROOT"/worktrees/*/ 2>/dev/null | head -1)}"; WT="${WT%/}"
COLOR_SRC="$CAP/images"; THERM_SRC="$CAP/thermal_images"
CALDIR="$WT/calibration"
OUT="$CALDIR/compute_out"; mkdir -p "$OUT"
HELD_C="$OUT/heldout_color"; HELD_T="$OUT/heldout_thermal"
echo "captures: $CAP"
echo "verifier: $WT"

# --- 1. select good pairs (BOTH SB-detect 12 corners) ------------------------
SEL="$OUT/selection.txt"
env -u PYTHONPATH "$PY" - "$COLOR_SRC" "$THERM_SRC" "$SEL" <<'PY' 2>&1 | grep -vE 'Deprecat|warn'
import cv2,glob,os,sys
CS,TS,SEL=sys.argv[1],sys.argv[2],sys.argv[3]
FL=cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY
def ok(g):
    for sz in [(4,3),(3,4)]:
        f,c=cv2.findChessboardCornersSB(g,sz,flags=FL)
        if f and c is not None and len(c)==12: return True
    return False
good=[]
for cp in sorted(glob.glob(f"{CS}/color_image_*.png"),key=lambda p:int(''.join(filter(str.isdigit,os.path.basename(p))))):
    i=int(''.join(filter(str.isdigit,os.path.basename(cp))))
    tp=f"{TS}/thermal_grayimage_{i}.png"
    cg=cv2.imread(cp,cv2.IMREAD_GRAYSCALE); tg=cv2.imread(tp,cv2.IMREAD_GRAYSCALE)
    if cg is None or tg is None: continue
    if ok(cg) and ok(tg): good.append(i)
open(SEL,"w").write("\n".join(map(str,good)))
print(f"good BOTH-detected pairs: {len(good)} -> {good}")
PY
mapfile -t GOOD < "$SEL"
NG=${#GOOD[@]}
if [ "$NG" -lt 22 ]; then echo "NEED >=22 good pairs (got $NG). Capture more / improve setup."; exit 1; fi
NFIT=$(( NG-12 )); [ "$NFIT" -gt 24 ] && NFIT=24
echo "using $NFIT fit + 12 held-out"

# --- 2. stage intrinsic + fit (fit pairs = first NFIT good) ------------------
rm -f "$CALDIR/thermal_images"/*.png "$CALDIR/color_images"/*.png 2>/dev/null
mkdir -p "$CALDIR/thermal_images" "$CALDIR/color_images" "$HELD_C" "$HELD_T"
rm -f "$HELD_C"/*.png "$HELD_T"/*.png 2>/dev/null
for k in $(seq 1 "$NFIT"); do
  src=${GOOD[$((k-1))]}
  cp "$THERM_SRC/thermal_grayimage_$src.png" "$CALDIR/thermal_images/thermal_grayimage_$k.png"
  cp "$COLOR_SRC/color_image_$src.png"       "$CALDIR/color_images/color_image_$k.png"
done
# held-out = next 12 good, named 25..36
for j in $(seq 0 11); do
  src=${GOOD[$((NFIT+j))]}; idx=$((25+j))
  cp "$COLOR_SRC/color_image_$src.png"       "$HELD_C/color_image_$idx.png"
  cp "$THERM_SRC/thermal_grayimage_$src.png" "$HELD_T/thermal_grayimage_$idx.png"
done

run_gui() { QT_QPA_PLATFORM=offscreen "$@"; }

# --- 3a. thermal intrinsics --------------------------------------------------
# Prefer a SEPARATE intrinsic-only thermal set (<captures>/intrinsic_thermal, from
# preview_capture --thermal-only) which can cover the whole thermal FOV. Fall back
# to the fit-pair thermal images if no dedicated set exists.
INTR_SRC="$CAP/intrinsic_thermal"
NINT=0
if ls "$INTR_SRC"/thermal_grayimage_*.png >/dev/null 2>&1; then
  echo "using dedicated intrinsic set: $INTR_SRC"
  rm -f "$CALDIR/thermal_images"/*.png 2>/dev/null
  NINT=0
  for f in $(ls "$INTR_SRC"/thermal_grayimage_*.png | sort -t_ -k3 -n); do
    NINT=$((NINT+1)); cp "$f" "$CALDIR/thermal_images/thermal_grayimage_$NINT.png"
  done
else
  echo "no intrinsic_thermal set; using fit-pair thermal for intrinsics (n=$NFIT)"
  NINT=$NFIT
fi
echo "=== camera_calibration (thermal intrinsics, n=$NINT) ==="
( cd "$CALDIR/build" && run_gui ./camera_calibration -r 4 -c 5 -n "$NINT" -pat 1 ) 2>&1 | tee "$OUT/camera_calibration.log" | tail -5
[ -f "$CALDIR/calibration.xml" ] || { echo "NO calibration.xml produced"; exit 1; }
# restore the fit-pair thermal images for the extrinsic step (intrinsics may have overwritten them)
rm -f "$CALDIR/thermal_images"/*.png 2>/dev/null
for k in $(seq 1 "$NFIT"); do
  src=${GOOD[$((k-1))]}
  cp "$THERM_SRC/thermal_grayimage_$src.png" "$CALDIR/thermal_images/thermal_grayimage_$k.png"
done

# --- 3b. extrinsics (resolve_extrinsic: mount-prior flip resolution) ----------
echo "=== resolve_extrinsic (thermal<->RGB, mount-prior flip, n=$NFIT) ==="
( cd "$CALDIR/build" && ./resolve_extrinsic \
    -c "$CALDIR/color_images" -t "$CALDIR/thermal_images" -n "$NFIT" \
    -i "$CALDIR/calibration.xml" -o "$CALDIR/extrinsic.xml" ) 2>&1 | tee "$OUT/extrinsic.log" | tail -12
[ -f "$CALDIR/extrinsic.xml" ] || { echo "NO extrinsic.xml produced"; exit 1; }

# --- 3c. held-out check (leakage-proof python; NEVER deletes a held-out pair) -
# Color-only front-facing IPPE pose (no thermal info), project board->color->thermal
# with the fixed extrinsic, marginalize ONLY SB's meaningless 180-deg thermal label
# (pick lower-RMS thermal order), report per-pair max/RMS + <=3.0px gate on all 12.
echo "=== held-out (12 disjoint pairs; color-only pose, no deletion) ==="
env -u PYTHONPATH "$PY" - "$HELD_C" "$HELD_T" "$CALDIR/calibration.xml" "$CALDIR/extrinsic.xml" "$RS_SERIAL" \
  > "$OUT/heldout.log" 2>&1 <<'PY'
import cv2,sys,glob,os,numpy as np, pyrealsense2 as rs
HELD_C,HELD_T,CALXML,EXTXML,SERIAL=sys.argv[1:6]
FL=cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY
obj=np.array([[c*0.03,r*0.03,0] for r in range(3) for c in range(4)],np.float32)
fs=cv2.FileStorage(CALXML,cv2.FILE_STORAGE_READ); Kt=fs.getNode("cameraMatrix").mat(); Dt=fs.getNode("distCoeffs").mat(); fs.release()
fe=cv2.FileStorage(EXTXML,cv2.FILE_STORAGE_READ); R=fe.getNode("R").mat(); T=fe.getNode("T").mat(); fe.release()
# color intrinsics live from the pinned D435i
p=rs.pipeline();cfg=rs.config();cfg.enable_device(SERIAL);cfg.enable_stream(rs.stream.color,1280,720,rs.format.rgb8,15)
pr=p.start(cfg);ci=pr.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics();p.stop()
Kc=np.array([[ci.fx,0,ci.ppx],[0,ci.fy,ci.ppy],[0,0,1]],float);Dc=np.zeros((5,1))
def sb(g):
    f,c=cv2.findChessboardCornersSB(g,(4,3),flags=FL)
    return c.reshape(-1,2).astype(np.float32) if (f and c is not None and len(c)==12) else None
def front_pose(pts):  # color-only: IPPE branches, keep front-facing, lowest reproj
    try: n,rvs,tvs,_=cv2.solvePnPGeneric(obj,pts,Kc,Dc,flags=cv2.SOLVEPNP_IPPE)
    except: n=0
    best=None
    for k in range(n):
        Rr,_=cv2.Rodrigues(rvs[k]);Xc=(Rr@obj.T+tvs[k]).T
        if np.all(Xc[:,2]>0):
            prj,_=cv2.projectPoints(obj,rvs[k],tvs[k],Kc,Dc);e=np.sqrt(((prj.reshape(-1,2)-pts)**2).sum(1)).mean()
            if best is None or e<best[0]: best=(e,Rr,tvs[k])
    if best is None:
        ok,rv,tv=cv2.solvePnP(obj,pts,Kc,Dc);Rr,_=cv2.Rodrigues(rv);best=(0,Rr,tv)
    return best[1],best[2]
cols=sorted(glob.glob(f"{HELD_C}/color_image_*.png"),key=lambda p:int(''.join(filter(str.isdigit,os.path.basename(p)))))
maxerrs=[]; fails=0
for cp in cols:
    i=int(''.join(filter(str.isdigit,os.path.basename(cp))))
    cg=cv2.imread(cp,cv2.IMREAD_GRAYSCALE); tg=cv2.imread(f"{HELD_T}/thermal_grayimage_{i}.png",cv2.IMREAD_GRAYSCALE)
    a=sb(tg); b=sb(cg)
    if a is None or b is None: print(f"  pair {i}: DETECT FAIL"); fails+=1; maxerrs.append(999); continue
    Rc,tc=front_pose(b)
    Xc=(Rc@obj.T+tc).T; Xt=(R.T@(Xc.T-T)).T
    proj,_=cv2.projectPoints(Xt.astype(np.float32),np.zeros(3),np.zeros(3),Kt,Dt); proj=proj.reshape(-1,2)
    er=np.sqrt(((proj-a)**2).sum(1)); ev=np.sqrt(((proj-a[::-1])**2).sum(1))
    e=er if er.mean()<=ev.mean() else ev
    maxerrs.append(float(e.max())); print(f"  pair {i}: max={e.max():.2f}px rms={np.sqrt((e**2).mean()):.2f}px")
m=np.array(maxerrs); gmax=m.max(); rms=np.sqrt((m[m<900]**2).mean()) if (m<900).any() else 999
print(f"HELD-OUT: {len(cols)} pairs, detect-fails={fails}, global_max={gmax:.2f}px RMS={rms:.2f}px")
print("GATE:", "PASS (max<=3.0 & RMS<=2.0)" if (gmax<=3.0 and rms<=2.0 and fails==0) else "FAIL")
PY
grep -vE 'Deprecat|warn' "$OUT/heldout.log"

# --- 4. report ---------------------------------------------------------------
echo; echo "============ RESULTS ============"
echo "intrinsics: $CALDIR/calibration.xml"
echo "extrinsics: $CALDIR/extrinsic.xml  (R,T thermal->color)"
echo "held-out log: $OUT/heldout.log"
grep -E 'HELD-OUT:|GATE:|\|T\|' "$OUT/extrinsic.log" "$OUT/heldout.log" 2>/dev/null | tail -6
