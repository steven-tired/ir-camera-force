// resolve_extrinsic — lean thermal↔RGB extrinsic with mount-prior flip resolution.
//
// Why this exists: findChessboardCornersSB (needed for the cut-out board) resolves the
// checkerboard's 180° orientation ambiguity INCONSISTENTLY between the 160x120 thermal and
// 1280x720 color views, so thermal[i] and color[i] are often not the same physical corner and
// a naive stereoCalibrate produces a garbage extrinsic (decimetre baseline). This tool fixes
// the per-pair flip using the known ~parallel Lepton↔D435i mount: for each pair it tries both
// thermal orderings, forms the candidate rig rotation G = Rc·Rtᵀ, and keeps the ordering whose
// G is closest to the current rig-rotation prior (identity on the first pass). It then drops
// pairs whose selected G is an outlier, runs stereoCalibrate, updates the prior from the fit,
// and iterates to a fixed point. Board poses use SOLVEPNP_IPPE with a front-facing/cheirality
// gate to reject the wrong planar-pose branch.
//
// FIT-only outlier rejection is legitimate; this tool is the accepted fit path. Held-out
// verification (heldout_verify) must NOT delete failing pairs.

#include <CalibrationContracts.h>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr char kSerial[] = "233522078685";
const int kSbFlags = cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY;

double geodesic_angle_deg(const cv::Mat &R)
{
    double t = (cv::trace(R)[0] - 1.0) / 2.0;
    t = std::max(-1.0, std::min(1.0, t));
    return std::acos(t) * 180.0 / CV_PI;
}

// SOLVEPNP_IPPE with a front-facing/cheirality gate: return the planar branch whose object
// points are all in front of the camera, with the lowest reprojection error. Falls back to
// ITERATIVE if IPPE yields no valid front-facing branch.
bool front_face_pose(const std::vector<cv::Point3f> &obj,
                     const std::vector<cv::Point2f> &pts,
                     const cv::Mat &K, const cv::Mat &D,
                     cv::Mat &R_out, cv::Mat &t_out)
{
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat reproj_err;
    double best_err = 1e18;
    bool found = false;
    try
    {
        int n = cv::solvePnPGeneric(obj, pts, K, D, rvecs, tvecs, false,
                                    cv::SOLVEPNP_IPPE, cv::noArray(), cv::noArray(), reproj_err);
        for (int k = 0; k < n; ++k)
        {
            cv::Mat Rk;
            cv::Rodrigues(rvecs[k], Rk);
            bool front = true;
            for (const auto &P : obj)
            {
                cv::Mat Xc = Rk * (cv::Mat_<double>(3, 1) << P.x, P.y, P.z) + tvecs[k];
                if (Xc.at<double>(2, 0) <= 0.0) { front = false; break; }
            }
            if (!front) continue;
            std::vector<cv::Point2f> pr;
            cv::projectPoints(obj, rvecs[k], tvecs[k], K, D, pr);
            double e = 0.0;
            for (size_t j = 0; j < pr.size(); ++j) e += cv::norm(pr[j] - pts[j]);
            e /= pr.size();
            if (e < best_err) { best_err = e; Rk.copyTo(R_out); tvecs[k].copyTo(t_out); found = true; }
        }
    }
    catch (const cv::Exception &) { found = false; }
    if (!found)
    {
        cv::Mat rv, tv;
        if (!cv::solvePnP(obj, pts, K, D, rv, tv)) return false;
        cv::Rodrigues(rv, R_out);
        tv.copyTo(t_out);
    }
    return true;
}

void usage(const char *p)
{
    std::cout << "Usage: " << p
              << " -c COLOR_DIR -t THERMAL_DIR -n N -i CALIBRATION_XML [-o EXTRINSIC_XML]"
                 " [-r ROWS] [-cc COLS] [--max-reject-frac F]\n"
                 "  Thermal intrinsics read from CALIBRATION_XML; color intrinsics from the\n"
                 "  live D435i (serial " << kSerial << "). Mount prior = identity (Lepton∥D435i).\n";
}

}  // namespace

int main(int argc, char **argv)
{
    std::string color_dir, thermal_dir, calib_xml, out_xml;
    int n = 0, rows = 4, cols = 5;
    double max_reject_frac = 0.15;
    for (int i = 1; i < argc; ++i)
    {
        auto need = [&](const char *f) { if (i + 1 >= argc) { std::cerr << "missing value for " << f << "\n"; std::exit(1); } return argv[++i]; };
        if (!std::strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else if (!std::strcmp(argv[i], "-c")) color_dir = need("-c");
        else if (!std::strcmp(argv[i], "-t")) thermal_dir = need("-t");
        else if (!std::strcmp(argv[i], "-n")) n = std::stoi(need("-n"));
        else if (!std::strcmp(argv[i], "-i")) calib_xml = need("-i");
        else if (!std::strcmp(argv[i], "-o")) out_xml = need("-o");
        else if (!std::strcmp(argv[i], "-r")) rows = std::stoi(need("-r"));
        else if (!std::strcmp(argv[i], "-cc")) cols = std::stoi(need("-cc"));
        else if (!std::strcmp(argv[i], "--max-reject-frac")) max_reject_frac = std::stod(need("--max-reject-frac"));
    }
    if (color_dir.empty() || thermal_dir.empty() || calib_xml.empty() || n <= 0)
    { usage(argv[0]); return 1; }
    if (out_xml.empty()) out_xml = thermal_dir + "/../extrinsic.xml";

    // thermal intrinsics from calibration.xml
    cv::Mat Kt, Dt;
    {
        cv::FileStorage fs(calib_xml, cv::FileStorage::READ);
        if (!fs.isOpened()) { std::cerr << "cannot open " << calib_xml << "\n"; return 1; }
        fs["cameraMatrix"] >> Kt;
        fs["distCoeffs"] >> Dt;
        if (Kt.empty() || Dt.empty()) { std::cerr << "calibration.xml missing cameraMatrix/distCoeffs\n"; return 1; }
    }

    // color intrinsics from the live D435i (pinned serial)
    cv::Mat Kc = cv::Mat::eye(3, 3, CV_64F), Dc = cv::Mat::zeros(5, 1, CV_64F);
    {
        rs2::pipeline pipe; rs2::config cfg;
        cfg.enable_device(kSerial);
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 15);
        try { pipe.start(cfg); }
        catch (const rs2::error &e) { std::cerr << "RealSense start failed: " << e.what() << "\n"; return 1; }
        auto vs = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        auto in = vs.get_intrinsics();
        if (!(in.model == RS2_DISTORTION_NONE || in.model == RS2_DISTORTION_BROWN_CONRADY ||
              (in.model == RS2_DISTORTION_INVERSE_BROWN_CONRADY &&
               in.coeffs[0] == 0 && in.coeffs[1] == 0 && in.coeffs[2] == 0 &&
               in.coeffs[3] == 0 && in.coeffs[4] == 0)))
        { std::cerr << "unsupported color distortion model with non-zero coeffs\n"; pipe.stop(); return 1; }
        Kc = (cv::Mat_<double>(3, 3) << in.fx, 0, in.ppx, 0, in.fy, in.ppy, 0, 0, 1);
        if (in.model == RS2_DISTORTION_BROWN_CONRADY)
            for (int k = 0; k < 5; ++k) Dc.at<double>(k, 0) = in.coeffs[k];
        pipe.stop();
    }

    const calibration::BoardContract board(rows, cols, 0.03);
    const cv::Size pattern = calibration::inner_corner_size(board);
    const std::vector<cv::Point3f> obj = calibration::object_points(board);

    // detect all pairs once
    struct Pair { std::vector<cv::Point2f> color, thermal; };
    std::vector<Pair> pairs;
    for (int idx = 1; idx <= n; ++idx)
    {
        calibration::CapturePair cp; std::string err;
        if (!calibration::load_capture_pair(color_dir, thermal_dir, idx, cp, err)) continue;
        std::vector<cv::Point2f> cc, tc;
        if (!cv::findChessboardCornersSB(cp.color_gray, pattern, cc, kSbFlags)) continue;
        if (!cv::findChessboardCornersSB(cp.thermal_gray, pattern, tc, kSbFlags)) continue;
        if (cc.size() != obj.size() || tc.size() != obj.size()) continue;
        pairs.push_back({cc, tc});
    }
    std::cout << "detected pairs: " << pairs.size() << " / " << n << "\n";
    if (!calibration::has_minimum_samples(pairs.size()))
    { std::cerr << "fewer than 10 detected pairs\n"; return 1; }

    // iterate: mount-prior flip selection + outlier filter + stereoCalibrate
    cv::Mat R_prior = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat R, T, E, F;
    std::vector<int> kept;
    for (int iter = 0; iter < 6; ++iter)
    {
        std::vector<std::vector<cv::Point2f>> selT, selC;
        std::vector<double> angs;
        std::vector<int> order;
        for (size_t p = 0; p < pairs.size(); ++p)
        {
            cv::Mat Rc, tc;
            if (!front_face_pose(obj, pairs[p].color, Kc, Dc, Rc, tc)) continue;
            double best_ang = 1e18; std::vector<cv::Point2f> best_t;
            for (int flip = 0; flip < 2; ++flip)
            {
                std::vector<cv::Point2f> tpts = pairs[p].thermal;
                if (flip) std::reverse(tpts.begin(), tpts.end());
                cv::Mat Rt, tt;
                if (!front_face_pose(obj, tpts, Kt, Dt, Rt, tt)) continue;
                cv::Mat G = Rc * Rt.t();
                double a = geodesic_angle_deg(R_prior.t() * G);
                if (a < best_ang) { best_ang = a; best_t = tpts; }
            }
            if (best_t.empty()) continue;
            angs.push_back(best_ang); selT.push_back(best_t); selC.push_back(pairs[p].color);
            order.push_back(static_cast<int>(p));
        }
        // median angle, keep pairs within max(8 deg, median+8 deg)
        std::vector<double> s = angs; std::sort(s.begin(), s.end());
        double med = s.empty() ? 0.0 : s[s.size() / 2];
        double gate = std::max(8.0, med + 8.0);
        std::vector<std::vector<cv::Point2f>> ft, fc; kept.clear();
        for (size_t k = 0; k < angs.size(); ++k)
            if (angs[k] <= gate) { ft.push_back(selT[k]); fc.push_back(selC[k]); kept.push_back(order[k]); }
        if (!calibration::has_minimum_samples(ft.size()))
        { std::cerr << "too few pairs after filtering\n"; return 1; }
        std::vector<std::vector<cv::Point3f>> objs(ft.size(), obj);
        cv::stereoCalibrate(objs, ft, fc, Kt, Dt, Kc, Dc, cv::Size(), R, T, E, F,
                            cv::CALIB_FIX_INTRINSIC);
        R.copyTo(R_prior);
        std::cout << "iter " << iter << ": kept " << ft.size() << "/" << pairs.size()
                  << "  rigRot " << geodesic_angle_deg(R) << " deg  |T| "
                  << cv::norm(T) * 100.0 << " cm\n";
    }

    double rejected_frac = 1.0 - static_cast<double>(kept.size()) / static_cast<double>(pairs.size());
    std::cout << "final rejected fraction: " << rejected_frac << " (limit " << max_reject_frac << ")\n";
    if (rejected_frac > max_reject_frac)
        std::cerr << "WARNING: rejection exceeds limit; dataset diversity/quality likely insufficient\n";
    if (!calibration::valid_calibration_values(R, T) || !calibration::valid_calibration_values(E, F))
    { std::cerr << "non-finite extrinsic\n"; return 1; }

    std::cout << "R:\n" << R << "\nT (m):\n" << T << "\n|T|: " << cv::norm(T) * 100.0 << " cm\n";
    cv::FileStorage fo(out_xml, cv::FileStorage::WRITE);
    if (!fo.isOpened()) { std::cerr << "cannot write " << out_xml << "\n"; return 1; }
    fo << "R" << R << "T" << T << "E" << E << "F" << F;
    fo.release();
    std::cout << "wrote " << out_xml << "\n";
    return 0;
}
