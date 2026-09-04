import pathlib
import subprocess
import sys
import tempfile


def opencv_xml(matrices: dict[str, tuple[int, int, list[float]]]) -> str:
    nodes = []
    for name, (rows, columns, values) in matrices.items():
        data = " ".join(str(value) for value in values)
        nodes.append(
            f'<{name} type_id="opencv-matrix">\n'
            f"  <rows>{rows}</rows><cols>{columns}</cols><dt>d</dt>\n"
            f"  <data>{data}</data>\n"
            f"</{name}>"
        )
    return '<?xml version="1.0"?>\n<opencv_storage>\n' + "\n".join(nodes) + "\n</opencv_storage>\n"


result = subprocess.run(
    [str(pathlib.Path(sys.argv[1])), "--help"],
    check=False,
    capture_output=True,
    text=True,
)
assert result.returncode == 0, result.stderr
for option in (
    "--color-dir",
    "--thermal-dir",
    "--intrinsic",
    "--extrinsic",
    "--output",
):
    assert option in result.stdout
for forbidden in (
    "--rows",
    "--columns",
    "--first-index",
    "--count",
    "--threshold-px",
    "--serial",
):
    assert forbidden not in result.stdout

with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    malformed = root / "malformed.xml"
    malformed.write_text("not OpenCV XML", encoding="utf-8")
    malformed_result = subprocess.run(
        [
            str(pathlib.Path(sys.argv[1])),
            "--color-dir",
            str(root),
            "--thermal-dir",
            str(root),
            "--intrinsic",
            str(malformed),
            "--extrinsic",
            str(malformed),
            "--output",
            str(root / "report.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert malformed_result.returncode == 1, malformed_result.stderr

    intrinsic = root / "intrinsic.xml"
    intrinsic.write_text(
        opencv_xml(
            {
                "cameraMatrix": (3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1]),
                "distCoeffs": (1, 5, [0, 0, 0, 0, 0]),
            }
        ),
        encoding="utf-8",
    )
    missing_ef = root / "missing-ef.xml"
    missing_ef.write_text(
        opencv_xml(
            {
                "R": (3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1]),
                "T": (3, 1, [0, 0, 0]),
            }
        ),
        encoding="utf-8",
    )
    missing_ef_result = subprocess.run(
        [
            str(pathlib.Path(sys.argv[1])),
            "--color-dir",
            str(root),
            "--thermal-dir",
            str(root),
            "--intrinsic",
            str(intrinsic),
            "--extrinsic",
            str(missing_ef),
            "--output",
            str(root / "missing-ef-report.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert missing_ef_result.returncode == 1, missing_ef_result.stderr
    assert "E and F" in missing_ef_result.stderr

    invalid_distortion = root / "invalid-distortion.xml"
    invalid_distortion.write_text(
        opencv_xml(
            {
                "cameraMatrix": (3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1]),
                "distCoeffs": (2, 3, [0, 0, 0, 0, 0, 0]),
            }
        ),
        encoding="utf-8",
    )
    valid_extrinsic = root / "valid-extrinsic.xml"
    valid_extrinsic.write_text(
        opencv_xml(
            {
                "R": (3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1]),
                "T": (3, 1, [0, 0, 0]),
                "E": (3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1]),
                "F": (3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1]),
            }
        ),
        encoding="utf-8",
    )
    invalid_distortion_result = subprocess.run(
        [
            str(pathlib.Path(sys.argv[1])),
            "--color-dir",
            str(root),
            "--thermal-dir",
            str(root),
            "--intrinsic",
            str(invalid_distortion),
            "--extrinsic",
            str(valid_extrinsic),
            "--output",
            str(root / "invalid-distortion-report.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert invalid_distortion_result.returncode == 1, invalid_distortion_result.stderr
    assert "invalid matrix shape" in invalid_distortion_result.stderr
