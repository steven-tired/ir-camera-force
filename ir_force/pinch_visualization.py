"""Lossless frame artifacts for descriptive direct-pinch visualization."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


EXPERIMENT_ROLE = "communication_only_descriptive_replication"
DECISION_AUTHORITY = "none"
AUTHORITATIVE_REFERENCE = "stage1e_tip_pinch_signal_05"


def _write_png_exclusive(path: Path, frame: np.ndarray) -> None:
    encoded, payload = cv2.imencode(".png", frame)
    if not encoded:
        raise RuntimeError(f"could not encode PNG {path}")
    with path.open("xb") as stream:
        stream.write(payload.tobytes())


class FrameArchive:
    """Archive each fresh Stage 1E attempt without changing its protocol."""

    _RAW_DIRS = {
        "thermal_uint16": Path("raw/thermal_uint16"),
        "d435_rgb": Path("raw/d435_rgb"),
        "d435_depth_z16": Path("raw/d435_depth_z16"),
    }

    def __init__(
        self,
        session_dir: Path,
        *,
        physical_protocol: str | None = None,
    ):
        self.session_dir = Path(session_dir)
        if physical_protocol not in (None, "contact_only"):
            raise ValueError("unsupported physical protocol")
        self.physical_protocol = physical_protocol
        if not self.session_dir.is_dir():
            raise FileNotFoundError(
                f"session directory does not exist: {self.session_dir}"
            )
        for relative_dir in self._RAW_DIRS.values():
            (self.session_dir / relative_dir).mkdir(
                parents=True,
                exist_ok=False,
            )

    def metadata(self) -> dict:
        metadata = {
            "experiment_id": self.session_dir.name,
            "experiment_role": EXPERIMENT_ROLE,
            "decision_authority": DECISION_AUTHORITY,
            "authoritative_reference": AUTHORITATIVE_REFERENCE,
            "can_update_thresholds": False,
            "can_authorize_stage1f": False,
            "artifact_paths": "relative_to_session_dir",
        }
        if self.physical_protocol == "contact_only":
            metadata.update(
                physical_protocol="contact_only",
                phase_semantics={
                    "record_just_touch": "light_contact",
                    "record_press_hard": (
                        "light_contact_legacy_analysis_slot"
                    ),
                    "record_return_touch": "light_contact",
                },
            )
        return metadata

    def capture(
        self,
        *,
        attempt_index: int,
        thermal_counts: np.ndarray,
        color_rgb: np.ndarray,
        depth_z16: np.ndarray,
    ) -> dict:
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
        ):
            raise ValueError("attempt_index must be a non-negative integer")
        if (
            not isinstance(thermal_counts, np.ndarray)
            or thermal_counts.shape != (120, 160)
            or thermal_counts.dtype != np.uint16
        ):
            raise ValueError(
                "thermal_counts must be a 120x160 uint16 array"
            )
        if (
            not isinstance(color_rgb, np.ndarray)
            or color_rgb.ndim != 3
            or color_rgb.shape[2] != 3
            or color_rgb.dtype != np.uint8
        ):
            raise ValueError("color_rgb must be an HxWx3 uint8 array")
        if (
            not isinstance(depth_z16, np.ndarray)
            or depth_z16.shape != color_rgb.shape[:2]
            or depth_z16.dtype != np.uint16
        ):
            raise ValueError(
                "depth_z16 must be uint16 and match the RGB image size"
            )

        stem = f"attempt_{attempt_index:06d}.png"
        relative_paths = {
            key: (relative_dir / stem).as_posix()
            for key, relative_dir in self._RAW_DIRS.items()
        }
        output_paths = {
            key: self.session_dir / relative_path
            for key, relative_path in relative_paths.items()
        }
        for path in output_paths.values():
            if path.exists():
                raise FileExistsError(path)

        _write_png_exclusive(
            output_paths["thermal_uint16"],
            thermal_counts,
        )
        _write_png_exclusive(
            output_paths["d435_rgb"],
            cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR),
        )
        _write_png_exclusive(
            output_paths["d435_depth_z16"],
            depth_z16,
        )
        return relative_paths


def _thermal_to_color(
    frame: np.ndarray,
    lower_count: float,
    upper_count: float,
) -> np.ndarray:
    if upper_count <= lower_count:
        upper_count = lower_count + 1.0
    scaled = np.clip(
        (frame.astype(np.float32) - lower_count)
        / (upper_count - lower_count),
        0.0,
        1.0,
    )
    u8 = np.rint(scaled * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def _percentile_range(frame: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(frame, (1.0, 99.0))
    lower = float(lower)
    upper = float(upper)
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def _artifact_path(session_dir: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("artifact path must be a string")
    path = (session_dir / relative_path).resolve()
    if not path.is_relative_to(session_dir.resolve()):
        raise ValueError("artifact path escapes session directory")
    return path


def _read_thermal_artifact(session_dir: Path, row: dict) -> np.ndarray:
    artifacts = row.get("frame_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("attempt is missing frame_artifacts")
    path = _artifact_path(session_dir, artifacts.get("thermal_uint16"))
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        frame is None
        or frame.shape != (120, 160)
        or frame.dtype != np.uint16
    ):
        raise ValueError(f"invalid thermal artifact: {path}")
    return frame


def _valid_fingertips(row: dict) -> list[tuple[str, int, int]]:
    if row.get("status") != "software_gate_accepted":
        return []
    tips = []
    for tip in row.get("fingertips", ()):
        if not isinstance(tip, dict):
            continue
        label = tip.get("label")
        pixel = tip.get("thermal_pixel")
        if (
            label not in ("thumb_tip", "index_tip")
            or not isinstance(pixel, (list, tuple))
            or len(pixel) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in pixel
            )
        ):
            continue
        x, y = pixel
        if 0 <= x < 160 and 0 <= y < 120:
            tips.append((label, x, y))
    return tips


def _annotate_fingertips(
    heatmap: np.ndarray,
    row: dict,
    *,
    scale: int,
) -> np.ndarray:
    annotated = heatmap.copy()
    colors = {
        "thumb_tip": (0, 255, 0),
        "index_tip": (255, 255, 0),
    }
    for label, x, y in _valid_fingertips(row):
        color = colors[label]
        center = (x * scale + scale // 2, y * scale + scale // 2)
        cv2.circle(annotated, center, max(3, scale), color, 1)
        cv2.rectangle(
            annotated,
            (
                max(0, (x - 1) * scale),
                max(0, (y - 1) * scale),
            ),
            (
                min(annotated.shape[1] - 1, (x + 2) * scale - 1),
                min(annotated.shape[0] - 1, (y + 2) * scale - 1),
            ),
            color,
            1,
        )
        cv2.putText(
            annotated,
            label,
            (min(center[0] + 4, annotated.shape[1] - 80), center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    cue = row.get("pinch_signal")
    if isinstance(cue, dict):
        text = (
            f"g{int(cue.get('group_index', 0)) + 1} "
            f"{cue.get('phase', 'unknown')}"
        )
        cv2.putText(
            annotated,
            text,
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _fingertip_crop(
    frame: np.ndarray,
    *,
    x: int,
    y: int,
    lower_count: float,
    upper_count: float,
) -> np.ndarray:
    half_width = 7
    native_size = half_width * 2 + 1
    padded = np.pad(frame, half_width, mode="edge")
    crop = padded[y : y + native_size, x : x + native_size]
    colored = _thermal_to_color(crop, lower_count, upper_count)
    crop_scale = 12
    rendered = cv2.resize(
        colored,
        (native_size * crop_scale, native_size * crop_scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.rectangle(
        rendered,
        (
            (half_width - 1) * crop_scale,
            (half_width - 1) * crop_scale,
        ),
        (
            (half_width + 2) * crop_scale - 1,
            (half_width + 2) * crop_scale - 1,
        ),
        (0, 255, 0),
        1,
    )
    cv2.circle(
        rendered,
        (
            half_width * crop_scale + crop_scale // 2,
            half_width * crop_scale + crop_scale // 2,
        ),
        4,
        (255, 255, 255),
        1,
    )
    return rendered


def render_session(capture_jsonl: Path, session_dir: Path) -> dict:
    """Render comparable heatmaps and fingertip crops from archived counts."""

    capture_jsonl = Path(capture_jsonl)
    session_dir = Path(session_dir)
    manifest_path = session_dir / "manifest.json"
    rendered_dir = session_dir / "rendered"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if rendered_dir.exists():
        raise FileExistsError(rendered_dir)

    rows = [
        json.loads(line)
        for line in capture_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata_rows = [
        row for row in rows if row.get("row_type") == "metadata"
    ]
    if len(metadata_rows) != 1:
        raise ValueError("capture must contain exactly one metadata row")
    identity = metadata_rows[0].get("visualization_capture")
    if (
        not isinstance(identity, dict)
        or identity.get("experiment_id") != session_dir.name
        or identity.get("experiment_role") != EXPERIMENT_ROLE
        or identity.get("decision_authority") != DECISION_AUTHORITY
    ):
        raise ValueError("visualization capture identity mismatch")

    attempts = [
        row
        for row in rows
        if row.get("row_type") == "attempt"
        and isinstance(row.get("frame_artifacts"), dict)
    ]
    if not attempts:
        raise ValueError("capture contains no archived attempts")
    attempt_frames = [
        (row, _read_thermal_artifact(session_dir, row))
        for row in attempts
    ]
    baseline = next(
        (
            (row, frame)
            for row, frame in attempt_frames
            if row.get("status") == "software_gate_accepted"
            and isinstance(row.get("pinch_signal"), dict)
            and row["pinch_signal"].get("phase") == "record_just_touch"
            and row["pinch_signal"].get("quota_accepted") is True
        ),
        None,
    )
    if baseline is None:
        baseline = attempt_frames[0]
        scale_source = "first_archived_frame_fallback"
    else:
        scale_source = "first_quota_accepted_record_just_touch"
    baseline_row, baseline_frame = baseline
    lower_count, upper_count = _percentile_range(baseline_frame)

    fixed_dir = rendered_dir / "fixed_scale"
    autoscale_dir = rendered_dir / "autoscale_display_only"
    overlay_dir = rendered_dir / "fingertip_overlays"
    crop_dirs = {
        label: rendered_dir / "fingertip_crops" / label
        for label in ("thumb_tip", "index_tip")
    }
    for path in (fixed_dir, autoscale_dir, overlay_dir, *crop_dirs.values()):
        path.mkdir(parents=True, exist_ok=False)

    full_frame_scale = 4
    overlay_count = 0
    crop_counts = {label: 0 for label in crop_dirs}
    for row, frame in attempt_frames:
        attempt_index = row.get("attempt_index")
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
        ):
            raise ValueError("invalid attempt index")
        filename = f"attempt_{attempt_index:06d}.png"
        fixed_native = _thermal_to_color(
            frame,
            lower_count,
            upper_count,
        )
        fixed = cv2.resize(
            fixed_native,
            (160 * full_frame_scale, 120 * full_frame_scale),
            interpolation=cv2.INTER_NEAREST,
        )
        _write_png_exclusive(fixed_dir / filename, fixed)

        auto_lower, auto_upper = _percentile_range(frame)
        autoscaled = cv2.resize(
            _thermal_to_color(frame, auto_lower, auto_upper),
            (160 * full_frame_scale, 120 * full_frame_scale),
            interpolation=cv2.INTER_NEAREST,
        )
        _write_png_exclusive(autoscale_dir / filename, autoscaled)

        tips = _valid_fingertips(row)
        if tips:
            _write_png_exclusive(
                overlay_dir / filename,
                _annotate_fingertips(
                    fixed,
                    row,
                    scale=full_frame_scale,
                ),
            )
            overlay_count += 1
        for label, x, y in tips:
            _write_png_exclusive(
                crop_dirs[label] / filename,
                _fingertip_crop(
                    frame,
                    x=x,
                    y=y,
                    lower_count=lower_count,
                    upper_count=upper_count,
                ),
            )
            crop_counts[label] += 1

    manifest = {
        "schema_version": 1,
        **identity,
        "capture_jsonl": capture_jsonl.relative_to(session_dir).as_posix(),
        "rendering": {
            "fixed_scale": {
                "comparison_allowed": True,
                "source": scale_source,
                "source_attempt_index": baseline_row["attempt_index"],
                "lower_count": lower_count,
                "upper_count": upper_count,
                "percentiles": [1.0, 99.0],
            },
            "autoscale_display_only": {
                "comparison_allowed": False,
                "scale": "per_frame_1st_99th_percentile",
            },
            "colormap": "inferno",
            "native_thermal_shape": [120, 160],
            "full_frame_scale": full_frame_scale,
        },
        "artifact_counts": {
            "archived_attempts": len(attempt_frames),
            "fixed_scale_heatmaps": len(attempt_frames),
            "autoscale_display_only_heatmaps": len(attempt_frames),
            "fingertip_overlays": overlay_count,
            "fingertip_crops": crop_counts,
        },
    }
    text = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.write("\n")
    return manifest
