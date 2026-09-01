from __future__ import annotations

import importlib


def test_default_layout_is_valid_for_the_frozen_thermal_constraints():
    module = importlib.import_module("ir_force.classifier.ir_foam_setup")

    regions = module.frozen_regions_from_layout(module.default_layout())

    assert regions.preflight_issues((128, 160)) == []


def test_thermal_drag_uses_display_scale_and_clamps_to_camera_frame():
    module = importlib.import_module("ir_force.classifier.ir_foam_setup")

    roi = module.roi_from_drag((312, 228), (404, 276), scale=4, frame_shape=(128, 160))
    clamped = module.roi_from_drag((-20, -20), (700, 600), scale=1, frame_shape=(480, 640))

    assert roi.as_list() == [78, 57, 24, 13]
    assert clamped.as_list() == [0, 0, 640, 480]


def test_recorder_arguments_follow_the_saved_layout():
    module = importlib.import_module("ir_force.classifier.ir_foam_setup")
    layout = module.default_layout()
    layout["oak_left_marker"] = module.PixelROI(200, 140, 30, 30)

    args = module.recorder_roi_arguments(layout)

    assert args[args.index("--thermal-foam-bbox") + 1] == "68,40,28,30"
    assert args[args.index("--oak-left-marker-roi") + 1] == "200,140,30,30"


def test_default_left_marker_roi_covers_the_verified_compression_path():
    module = importlib.import_module("ir_force.classifier.ir_foam_setup")

    roi = module.default_layout()["oak_left_marker"]
    for x, y in ((235, 161), (281, 155)):
        assert roi.x <= x < roi.x_end
        assert roi.y <= y < roi.y_end
