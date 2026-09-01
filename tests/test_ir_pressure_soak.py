import ast
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "ir_pressure_soak.py"
PROJECT_ROOT = MODULE_PATH.parent
ROBOT_FREE_ENV = "LEROBOT_TELEOPERATOR_SO101_WEBCAM_ROBOT_FREE_IMPORT"
#: What a robot-free soak import is allowed to pull in. Before the split these
#: modules lived under the `lerobot_teleoperator_so101_webcam` package, so a
#: `startswith("lerobot")` probe caught them; they are now `ir_force.*` and the
#: probe must name both prefixes or it silently asserts nothing.
ROBOT_FREE_MODULES = [
    "ir_force",
    "ir_force.ir_hand_calibration",
    "ir_force.ir_hand_roi",
    "ir_force.ir_pressure",
    "ir_force.ir_pressure_proposal",
    "ir_force.ir_shadow_telemetry",
]


def _load_module():
    assert MODULE_PATH.exists(), "ir_pressure_soak.py has not been implemented"
    spec = importlib.util.spec_from_file_location("ir_pressure_soak", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_requires_sidecar_and_accepts_defaults_and_zero_test_overrides():
    module = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args([])

    defaults = module.parse_args(["--sidecar", "soak.csv"])
    assert defaults.sidecar == "soak.csv"
    assert defaults.duration_s == 1800.0
    assert defaults.max_oak_stall_ms == 500.0
    assert defaults.min_cycles == 100
    assert defaults.thermal == "/dev/video21"
    assert Path(defaults.calibration).name == "oak_flir_hand_pressure_projection.json"

    zero = module.parse_args(
        [
            "--sidecar",
            "smoke.csv",
            "--duration-s",
            "0",
            "--min-cycles",
            "0",
        ]
    )
    assert zero.duration_s == 0.0
    assert zero.min_cycles == 0


_ALLOWED_MODULE_IMPORTS = {
    "argparse",
    "cv2",
    "json",
    "math",
    "numpy",
    "os",
    "sys",
    "threading",
    "time",
    "webcam_input",
}
_ALLOWED_FROM_IMPORTS = {
    "__future__": {"annotations"},
    "collections": {"Counter"},
    "collections.abc": {"Callable"},
    "dataclasses": {"dataclass", "field"},
    "pathlib": {"Path"},
    "ir_force.ir_pressure": {
        "HandPressureEstimator",
        "PressureConfig",
        "inactive_pressure",
        "timing_limit_exceeded",
    },
    "ir_force.ir_hand_calibration": {
        "load_projection_calibration",
        "validate_projection_calibration",
    },
    "ir_force.ir_pressure_proposal": {
        "GRIP_CLOSE_ALPHA",
        "GRIP_OPEN_ALPHA",
        "GRIP_OVERDRIVE",
        "PressureProposalStateMachine",
        "apply_pressure_overdrive",
    },
    "ir_force.ir_shadow_telemetry": {
        "IRShadowTelemetryLogger",
        "IRShadowTelemetrySample",
    },
    "webcam_input.depth": {"ScaleDepthStrategy"},
    "webcam_input.webcam_source": {"WebcamSource"},
    "webcam_input.wrist_estimator": {"WebcamWristEstimator"},
}
_DYNAMIC_IMPORT_ROOTS = {"importlib", "pkgutil", "runpy", "zipimport"}
_DYNAMIC_LOADER_NAMES = {
    "ExtensionFileLoader",
    "FileFinder",
    "PathFinder",
    "SourceFileLoader",
    "SourcelessFileLoader",
    "__import__",
    "exec_module",
    "find_loader",
    "find_spec",
    "get_loader",
    "import_module",
    "load_module",
    "module_from_spec",
    "spec_from_file_location",
}
_CODE_EXEC_NAMES = {"compile", "eval", "exec"}
_DYNAMIC_CAPABILITY_NAMES = _DYNAMIC_LOADER_NAMES.union(_CODE_EXEC_NAMES)
_DANGEROUS_NAMESPACE_NAMES = {"__builtins__", "__loader__"}
_DANGEROUS_MAPPING_ATTRIBUTES = {"__builtins__", "__dict__"}
_DANGEROUS_NAMESPACE_ATTRIBUTES = _DANGEROUS_MAPPING_ATTRIBUTES.union({"modules"})


def _call_leaf(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _constant_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_obvious_capability_mapping(node):
    if isinstance(node, ast.Name):
        return node.id == "__builtins__"
    if isinstance(node, ast.Attribute):
        return node.attr in _DANGEROUS_MAPPING_ATTRIBUTES
    if isinstance(node, ast.Subscript):
        return _constant_string(node.slice) == "__builtins__"
    if isinstance(node, ast.Call):
        return _call_leaf(node.func) in {"globals", "vars"}
    return False


def _record_dynamic_lookup(node, violations):
    leaf = _call_leaf(node.func)
    if leaf == "getattr" and len(node.args) >= 2:
        lookup_name = _constant_string(node.args[1])
    elif (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "dict"
        and leaf in {"get", "__getitem__"}
        and len(node.args) >= 2
    ):
        lookup_name = _constant_string(node.args[1])
    elif (
        isinstance(node.func, ast.Attribute)
        and leaf in {"get", "__getitem__"}
        and node.args
        and _is_obvious_capability_mapping(node.func.value)
    ):
        lookup_name = _constant_string(node.args[0])
    else:
        return
    if lookup_name in _DYNAMIC_CAPABILITY_NAMES:
        violations.add((node.lineno, f"dynamic capability lookup: {lookup_name}"))


def _find_prohibited_import_behaviors(source, *, filename="<source>"):
    tree = ast.parse(source, filename=filename)
    violations = set()

    # B2b imports must be added to these allowlists deliberately.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name not in _ALLOWED_MODULE_IMPORTS:
                    violations.add(
                        (node.lineno, f"prohibited import: {imported.name}")
                    )
                root = imported.name.split(".")[0]
                if root in _DYNAMIC_IMPORT_ROOTS:
                    violations.add((node.lineno, f"dynamic import module: {imported.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            allowed_symbols = _ALLOWED_FROM_IMPORTS.get(module, set())
            root = module.split(".")[0]
            if root in _DYNAMIC_IMPORT_ROOTS:
                violations.add((node.lineno, f"dynamic import module: {module}"))
            for imported in node.names:
                full_name = ".".join(part for part in (module, imported.name) if part)
                if node.level or imported.name not in allowed_symbols:
                    violations.add((node.lineno, f"prohibited import: {full_name}"))
                if imported.name in _DYNAMIC_LOADER_NAMES:
                    violations.add((node.lineno, f"dynamic loader import: {full_name}"))
                elif imported.name in _CODE_EXEC_NAMES:
                    violations.add(
                        (node.lineno, f"dynamic code execution import: {full_name}")
                    )
        elif isinstance(node, ast.Call):
            leaf = _call_leaf(node.func)
            if leaf in _DYNAMIC_LOADER_NAMES:
                violations.add((node.lineno, f"dynamic loader call: {leaf}"))
            if leaf in _CODE_EXEC_NAMES:
                violations.add((node.lineno, f"dynamic code execution: {leaf}"))
            _record_dynamic_lookup(node, violations)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _DANGEROUS_NAMESPACE_NAMES:
                violations.add(
                    (node.lineno, f"dynamic namespace reference: {node.id}")
                )
            if node.id in _DYNAMIC_LOADER_NAMES:
                violations.add((node.lineno, f"dynamic loader reference: {node.id}"))
            if node.id in _CODE_EXEC_NAMES:
                violations.add(
                    (node.lineno, f"dynamic code execution reference: {node.id}")
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in _DANGEROUS_NAMESPACE_ATTRIBUTES:
                violations.add(
                    (node.lineno, f"dynamic namespace reference: {node.attr}")
                )
            if node.attr in _DYNAMIC_LOADER_NAMES:
                violations.add(
                    (node.lineno, f"dynamic loader reference: {node.attr}")
                )
            if node.attr in _CODE_EXEC_NAMES:
                violations.add(
                    (node.lineno, f"dynamic code execution reference: {node.attr}")
                )
        elif (
            isinstance(node, ast.Subscript)
            and _is_obvious_capability_mapping(node.value)
            and _constant_string(node.slice) in _DYNAMIC_CAPABILITY_NAMES
        ):
            violations.add(
                (
                    node.lineno,
                    f"dynamic capability lookup: {_constant_string(node.slice)}",
                )
            )

    return [message for _, message in sorted(violations)]


def test_ast_and_imports_prohibit_robot_or_live_teleop_references():
    assert MODULE_PATH.exists(), "ir_pressure_soak.py has not been implemented"
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert _find_prohibited_import_behaviors(source, filename=str(MODULE_PATH)) == []


_REVIEW_PROBES = [
    (
        "review_module_dunder_builtins",
        "import os\nos.__builtins__['__import__']('serial')",
        "dynamic capability lookup",
    ),
    (
        "review_default_plugin_root",
        "import lerobot_teleoperator_so101_webcam as plugin\n"
        "plugin.SO101WebcamConfig()",
        "prohibited import",
    ),
    (
        "review_unbound_dict_getitem",
        "import builtins\n"
        "dict.__getitem__(builtins.__dict__, '__import__')('serial')",
        "dynamic capability lookup",
    ),
    (
        "review_unbound_dict_get",
        "import sys\n"
        "dict.get(sys.modules['importlib'].__dict__, 'import_module')('serial')",
        "dynamic capability lookup",
    ),
    (
        "review_unlisted_stdlib_module",
        "import statistics",
        "prohibited import",
    ),
    (
        "review_unlisted_allowed_module_symbol",
        "from pathlib import PurePath",
        "prohibited import",
    ),
    (
        "review_so101_webcam_config",
        "from lerobot_teleoperator_so101_webcam.config_so101_webcam "
        "import SO101WebcamConfig",
        "prohibited import",
    ),
    (
        "review_so101_webcam_ee_config",
        "import lerobot_teleoperator_so101_webcam.config_so101_webcam_ee",
        "prohibited import",
    ),
    (
        "review_so101_webcam_ee_config_class",
        "from plugin_api import SO101WebcamEEConfig",
        "prohibited import",
    ),
    (
        "review_robot_config_class",
        "from plugin_api import RobotConfig",
        "prohibited import",
    ),
    (
        "review_follower_config_class",
        "from plugin_api import SO101FollowerConfig",
        "prohibited import",
    ),
    (
        "review_generic_follower_config_class",
        "from plugin_api import FollowerConfig",
        "prohibited import",
    ),
    (
        "review_sys_modules_loader",
        "import sys\nsys.modules['importlib'].import_module('serial')",
        "dynamic loader",
    ),
    (
        "review_sys_modules_get_loader",
        "import sys\nsys.modules.get('importlib').import_module('serial')",
        "dynamic loader",
    ),
    (
        "review_loader_constructor_chain",
        "type(__loader__)('dynamic', '/tmp/dynamic.py').load_module('dynamic')",
        "dynamic loader",
    ),
    (
        "review_spec_loader_chain",
        "get_spec().loader.exec_module(module)",
        "dynamic loader",
    ),
    (
        "review_loader_dict_get",
        "type(__loader__).__dict__.get('load_module')('dynamic')",
        "dynamic capability lookup",
    ),
    (
        "review_late_builtin_rebinding",
        "exec('import serial')\nexec = print",
        "dynamic code execution",
    ),
    (
        "review_builtins_dict_get",
        "import builtins\nbuiltins.__dict__.get('__import__')('serial')",
        "dynamic capability lookup",
    ),
    (
        "review_builtins_dict_subscript",
        "import builtins\nbuiltins.__dict__['__import__']('serial')",
        "dynamic capability lookup",
    ),
    (
        "review_dunder_builtins_get",
        "__builtins__.get('__import__')('serial')",
        "dynamic capability lookup",
    ),
    (
        "review_dunder_builtins_subscript",
        "__builtins__['__import__']('serial')",
        "dynamic capability lookup",
    ),
    (
        "review_globals_builtins_get",
        "globals()['__builtins__'].get('__import__')('serial')",
        "dynamic capability lookup",
    ),
]


@pytest.mark.parametrize(
    ("name", "source", "expected_violation"),
    [
        ("direct_robot", "import lerobot.robots", "prohibited import"),
        ("direct_teleop", "import teleop_viz_ee", "prohibited import"),
        ("direct_motor", "import lerobot.motors", "prohibited import"),
        ("direct_serial", "import serial", "prohibited import"),
        ("parent_robot", "from lerobot import robots as r", "prohibited import"),
        (
            "parent_teleop",
            "from lerobot import teleoperators as t",
            "prohibited import",
        ),
        ("parent_motor", "from lerobot import motors as m", "prohibited import"),
        (
            "from_import_serial",
            "from device_support import serial as driver",
            "prohibited import",
        ),
        (
            "function_local",
            "def deferred():\n    from lerobot import robots as r\n    return r",
            "prohibited import",
        ),
        (
            "importlib_alias",
            "import importlib as il\nil.util.spec_from_file_location('r', '/tmp/r.py')",
            "dynamic import module",
        ),
        (
            "loader_alias",
            "from importlib.util import spec_from_file_location as locate\n"
            "locate('r', '/tmp/r.py')",
            "dynamic loader import",
        ),
        (
            "code_executor_alias",
            "run = exec\nrun(\"import serial\")",
            "dynamic code execution",
        ),
        (
            "builtin_lookup",
            "import builtins\ngetattr(builtins, '__import__')('serial')",
            "dynamic capability lookup",
        ),
        ("exec_string", "exec(\"import serial\")", "dynamic code execution"),
        (
            "eval_string",
            "eval(\"__import__('lerobot.robots')\")",
            "dynamic code execution",
        ),
        (
            "compile_string",
            "compile('from lerobot import motors', '<dynamic>', 'exec')",
            "dynamic code execution",
        ),
    ]
    + _REVIEW_PROBES,
)
def test_source_checker_rejects_deferred_and_indirect_imports(
    name, source, expected_violation
):
    violations = _find_prohibited_import_behaviors(source)

    assert any(expected_violation in violation for violation in violations), name


def test_source_checker_ignores_benign_documentation_strings():
    source = '''
"""Examples mention `from lerobot import robots` and `exec("import serial")`."""

# Documentation may also say: import lerobot.motors
EXAMPLE = "compile('from lerobot import motors', '<dynamic>', 'exec')"

from ir_force.ir_pressure import PressureConfig
'''

    assert _find_prohibited_import_behaviors(source) == []


def test_source_checker_allows_explicit_sys_import_and_stderr():
    source = "import sys\nprint('soak failed', file=sys.stderr)"

    assert _find_prohibited_import_behaviors(source) == []


PUBLIC_REPO = PROJECT_ROOT.parent / "mediapipe-so101"
PUBLIC_WEBCAM_INPUT_SRC = PUBLIC_REPO / "packages" / "webcam_input" / "src"


def _public_package_paths():
    """Where the public mediapipe-so101 packages are, for the subprocess.

    Before the repository split, `webcam_input` sat in this same checkout, so
    clearing PYTHONPATH and running from PROJECT_ROOT was enough. It is now a
    separate package, and this environment still carries an *editable install
    pointing at the pre-split `webcam-input/` tree* -- which `ir_pressure_soak`
    refuses to load. So point the child at the public checkout explicitly:
    PYTHONPATH precedes site-packages, so it wins over the stale .pth entry.
    """
    return [str(PUBLIC_WEBCAM_INPUT_SRC)] if PUBLIC_WEBCAM_INPUT_SRC.is_dir() else []


def _fresh_python(code):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop(ROBOT_FREE_ENV, None)
    public = _public_package_paths()
    if public:
        environment["PYTHONPATH"] = os.pathsep.join(public)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_fresh_import_of_soak_keeps_robot_and_teleoperator_modules_unloaded():
    result = _fresh_python(
        f"""
import json
import os
import sys
import ir_pressure_soak
import webcam_input

prohibited = sorted(
    name for name in sys.modules
    if name == "serial"
    or name == "ir_force.ir_robot"
    or name.startswith("serial.")
    or name.startswith("lerobot.teleoperators")
    or name.startswith("lerobot.robots")
    or name.startswith("lerobot.motors")
    or ".motors." in name
    or name.endswith("teleop_viz_ee")
    or name.endswith("record_so101_ee")
    or name.endswith("config_so101_webcam")
    or name.endswith("config_so101_webcam_ee")
    or name.endswith(".so101_webcam")
    or name.endswith(".so101_webcam_ee")
    or ".bus" in name.lower()
    or ".action" in name.lower()
    or ".observation" in name.lower()
)
print(json.dumps({{
    "prohibited": prohibited,
    "env": os.environ.get({ROBOT_FREE_ENV!r}),
    "webcam_input": webcam_input.__file__,
    "soak_modules": sorted(name for name in sys.modules
                       if name.startswith("lerobot") or name.startswith("ir_force")),
}}))
"""
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "prohibited": [],
        "env": None,
        "webcam_input": str(PUBLIC_WEBCAM_INPUT_SRC / "webcam_input" / "__init__.py"),
        "soak_modules": ROBOT_FREE_MODULES,
    }


def test_fresh_default_package_import_still_exposes_plugin_classes():
    result = _fresh_python(
        """
import json
import lerobot_teleoperator_so101_webcam as plugin

print(json.dumps({
    "all": plugin.__all__,
    "classes": [
        plugin.SO101WebcamConfig.__name__,
        plugin.SO101Webcam.__name__,
        plugin.SO101WebcamEEConfig.__name__,
        plugin.SO101WebcamEE.__name__,
    ],
}))
"""
    )
    payload = json.loads(result.stdout)

    assert payload["all"] == [
        "SO101WebcamConfig",
        "SO101Webcam",
        "SO101WebcamEEConfig",
        "SO101WebcamEE",
    ]
    assert payload["classes"] == [
        "SO101WebcamConfig",
        "SO101Webcam",
        "SO101WebcamEEConfig",
        "SO101WebcamEE",
    ]


class ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeSource:
    def __init__(self, samples, *, oak_failed=False, error=None):
        self.samples = list(samples)
        self.index = 0
        self.oak_failed = oak_failed
        self.error = error
        self.calls = 0

    def latest_sample(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        sample = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        return sample


class FakeEstimator:
    def __init__(self, readings, *, update_error=None, reset_error=None):
        self.readings = list(readings)
        self.update_error = update_error
        self.reset_error = reset_error
        self.update_calls = 0
        self.reset_calls = 0

    def update(self, landmarks, pinch, enabled):
        self.update_calls += 1
        if self.update_error is not None:
            raise self.update_error
        assert landmarks.valid
        assert enabled is True
        return self.readings.pop(0)

    def reset(self):
        self.reset_calls += 1
        if self.reset_error is not None:
            raise self.reset_error


class FakeLogger:
    path = Path("fake-sidecar.csv")

    def __init__(
        self,
        *,
        enabled=True,
        disable_after=None,
        finalize_error=None,
        clock=None,
        finalize_delays=(),
    ):
        self.enabled = enabled
        self.disable_after = disable_after
        self.finalize_error = finalize_error
        self.clock = clock
        self.finalize_delays = list(finalize_delays)
        self.rows = []

    def finalize(self, sample, *, command_sent):
        if self.finalize_error is not None:
            raise self.finalize_error
        self.rows.append((sample, command_sent))
        if self.finalize_delays:
            self.clock.now += self.finalize_delays.pop(0)
        if self.disable_after is not None and len(self.rows) >= self.disable_after:
            self.enabled = False


def _sample(
    frame_id,
    observed_at_s,
    pinch,
    *,
    landmarks_valid=True,
    wrist_valid=True,
    fist_state="open",
):
    points = np.zeros((21, 3), dtype=float)
    points[4, 0] = pinch
    landmarks = SimpleNamespace(
        landmarks=points,
        valid=landmarks_valid,
        observed_at_s=observed_at_s,
        frame_id=frame_id,
    )
    wrist = SimpleNamespace(valid=wrist_valid, fist_state=fist_state)
    return SimpleNamespace(
        wrist=wrist,
        landmarks=landmarks,
        observed_at_s=observed_at_s,
        frame_id=frame_id,
    )


def _no_publication():
    return _sample(
        None,
        None,
        0.0,
        landmarks_valid=False,
        wrist_valid=False,
    )


def _pressure(
    status,
    *,
    active,
    available=True,
    oak_age_s=0.0,
    thermal_age_s=0.0,
    sensor_skew_s=0.0,
):
    return SimpleNamespace(
        pressure_0_1=0.5 if active else 0.0,
        active=active,
        quality=1.0 if available else 0.0,
        available=available,
        status=status,
        roi=None,
        roi_mode="projected_fingertips",
        oak_observed_at_s=0.0,
        thermal_observed_at_s=0.0,
        sensor_skew_s=sensor_skew_s,
        oak_age_s=oak_age_s,
        thermal_age_s=thermal_age_s,
    )


def _assert_metric(metric, *, count, p50, p95, p99, maximum):
    assert metric["count"] == count
    assert metric["p50"] == pytest.approx(p50) if p50 is not None else metric["p50"] is None
    assert metric["p95"] == pytest.approx(p95) if p95 is not None else metric["p95"] is None
    assert metric["p99"] == pytest.approx(p99) if p99 is not None else metric["p99"] is None
    assert metric["max"] == pytest.approx(maximum) if maximum is not None else metric["max"] is None


def test_run_soak_counts_open_closed_open_cycle_and_never_marks_commands_sent():
    module = _load_module()
    clock = ManualClock()
    source = FakeSource(
        [
            _sample(0, 0.00, 0.08),
            _sample(1, 0.01, 0.04),
            _sample(2, 0.02, 0.08),
        ]
    )
    estimator = FakeEstimator(
        [
            _pressure("baseline", active=False),
            _pressure("active", active=True),
            _pressure("baseline", active=False),
        ]
    )
    logger = FakeLogger()

    summary = module.run_soak(
        source=source,
        estimator=estimator,
        logger=logger,
        duration_s=0.025,
        min_cycles=1,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert isinstance(summary, module.SoakSummary)
    assert summary.exit_code == 0
    assert summary.reason == "completed"
    assert summary.ticks == 3
    assert summary.cycles == 1
    assert summary.state_counts == {"MOVING": 3}
    assert summary.status_counts == {"active": 1, "baseline": 2}
    assert summary.rejection_counts == {}
    assert summary.fault_closure_violations == 0
    assert summary.sidecar == "fake-sidecar.csv"
    assert len(logger.rows) == 3
    assert all(command_sent is False for _, command_sent in logger.rows)

    rows = [sample for sample, _ in logger.rows]
    assert [row.base_gripper_pos for row in rows] == pytest.approx([60.0, 20.0, 60.0])
    assert [row.proposed_gripper_pos for row in rows] == pytest.approx(
        [52.0, 50.6, 50.81]
    )
    assert [row.actual_gripper_pos for row in rows] == pytest.approx([42.0, 14.0, 18.2])
    assert [row.baseline_ready for row in rows] == [True, True, True]
    assert [row.fault_latched for row in rows] == [False, False, False]
    _assert_metric(
        summary.metrics["oak_age_ms"],
        count=3,
        p50=0.0,
        p95=0.0,
        p99=0.0,
        maximum=0.0,
    )
    _assert_metric(
        summary.metrics["thermal_age_ms"],
        count=3,
        p50=0.0,
        p95=0.0,
        p99=0.0,
        maximum=0.0,
    )
    _assert_metric(
        summary.metrics["pair_skew_ms"],
        count=3,
        p50=0.0,
        p95=0.0,
        p99=0.0,
        maximum=0.0,
    )
    _assert_metric(
        summary.metrics["loop_period_ms"],
        count=2,
        p50=10.0,
        p95=10.0,
        p99=10.0,
        maximum=10.0,
    )
    _assert_metric(
        summary.metrics["control_latency_ms"],
        count=3,
        p50=0.0,
        p95=0.0,
        p99=0.0,
        maximum=0.0,
    )


def test_run_soak_hold_skips_flir_and_retains_legacy_actual_gripper():
    module = _load_module()
    clock = ManualClock()
    estimator = FakeEstimator([_pressure("baseline", active=False)])
    logger = FakeLogger()

    summary = module.run_soak(
        source=FakeSource(
            [
                _sample(0, 0.00, 0.08),
                _sample(1, 0.01, 0.0, wrist_valid=False),
            ]
        ),
        estimator=estimator,
        logger=logger,
        duration_s=0.015,
        min_cycles=0,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.exit_code == 0
    assert summary.state_counts == {"HOLD": 1, "MOVING": 1}
    assert summary.status_counts == {"baseline": 1, "hold": 1}
    assert estimator.update_calls == 1
    assert estimator.reset_calls == 1
    moving, held = [row for row, _ in logger.rows]
    assert moving.actual_gripper_pos == pytest.approx(42.0)
    assert held.state == "HOLD"
    assert held.pressure is None
    assert held.proposed_gripper_pos == pytest.approx(52.0)
    assert held.actual_gripper_pos == pytest.approx(42.0)
    assert held.fallback_reason == "hold"


def test_run_soak_middle_skips_flir_resets_and_uses_exact_middle_actual():
    module = _load_module()
    clock = ManualClock()
    estimator = FakeEstimator([])
    logger = FakeLogger()

    summary = module.run_soak(
        source=FakeSource([_sample(0, 0.0, 0.08, fist_state="closed")]),
        estimator=estimator,
        logger=logger,
        duration_s=0.005,
        min_cycles=0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert summary.exit_code == 0
    assert summary.state_counts == {"MIDDLE": 1}
    assert summary.status_counts == {"middle": 1}
    assert estimator.update_calls == 0
    assert estimator.reset_calls == 1
    row, command_sent = logger.rows[0]
    assert command_sent is False
    assert row.state == "MIDDLE"
    assert row.pressure is None
    assert row.base_gripper_pos == pytest.approx(60.0)
    assert row.proposed_gripper_pos == pytest.approx(50.0)
    assert row.actual_gripper_pos == pytest.approx(50.0)
    assert row.fallback_reason == "middle"


@pytest.mark.parametrize(
    ("fist_state", "remove_fist_state"),
    [
        pytest.param("unknown", False, id="unknown"),
        pytest.param(None, False, id="none"),
        pytest.param(None, True, id="missing"),
        pytest.param([], False, id="unexpected-type"),
    ],
)
def test_run_soak_unknown_clutch_holds_and_skips_flir(
    fist_state, remove_fist_state
):
    module = _load_module()
    clock = ManualClock()
    sample = _sample(0, 0.0, 0.08, fist_state=fist_state)
    if remove_fist_state:
        del sample.wrist.fist_state
    estimator = FakeEstimator([])
    logger = FakeLogger()

    summary = module.run_soak(
        source=FakeSource([sample]),
        estimator=estimator,
        logger=logger,
        duration_s=0.005,
        min_cycles=0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert summary.exit_code == 0
    assert summary.state_counts == {"HOLD": 1}
    assert summary.status_counts == {"clutch_unknown": 1}
    assert estimator.update_calls == 0
    assert estimator.reset_calls == 1
    row, command_sent = logger.rows[0]
    assert command_sent is False
    assert row.state == "HOLD"
    assert row.pressure is None
    assert row.fallback_reason == "clutch_unknown"


def test_run_soak_counts_exactly_one_cycle_through_near_threshold_jitter():
    module = _load_module()
    clock = ManualClock()
    pinches = [0.08, 0.05, 0.04, 0.05, 0.08, 0.08]
    source = FakeSource(
        [_sample(index, index * 0.01, pinch) for index, pinch in enumerate(pinches)]
    )
    estimator = FakeEstimator([_pressure("baseline", active=False) for _ in pinches])
    logger = FakeLogger()

    summary = module.run_soak(
        source=source,
        estimator=estimator,
        logger=logger,
        duration_s=0.059,
        min_cycles=1,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.exit_code == 0
    assert summary.ticks == 6
    assert summary.cycles == 1


def test_run_soak_exits_when_no_first_oak_publication_arrives():
    module = _load_module()
    clock = ManualClock()
    source = FakeSource([_no_publication()])
    estimator = FakeEstimator([])
    logger = FakeLogger()

    summary = module.run_soak(
        source=source,
        estimator=estimator,
        logger=logger,
        duration_s=1.0,
        min_cycles=0,
        max_oak_stall_s=0.02,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.exit_code != 0
    assert summary.reason == "oak_no_first_frame_stall"
    assert summary.ticks == 0
    assert source.calls == 4
    assert estimator.update_calls == 0
    assert logger.rows == []


def test_run_soak_checks_no_first_publication_before_duration_completion():
    module = _load_module()
    clock = ManualClock()
    source = FakeSource([_no_publication()])

    summary = module.run_soak(
        source=source,
        estimator=FakeEstimator([]),
        logger=FakeLogger(),
        duration_s=0.51,
        min_cycles=0,
        max_oak_stall_s=0.5,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.51,
    )

    assert summary.exit_code != 0
    assert summary.reason == "oak_no_first_frame_stall"
    assert summary.ticks == 0
    assert source.calls == 2


def test_run_soak_repeated_frame_id_is_not_watchdog_progress():
    module = _load_module()
    clock = ManualClock()
    source = FakeSource([_sample(7, 0.0, 0.08)])
    estimator = FakeEstimator(
        [_pressure("baseline", active=False) for _ in range(3)]
    )
    logger = FakeLogger()

    summary = module.run_soak(
        source=source,
        estimator=estimator,
        logger=logger,
        duration_s=1.0,
        min_cycles=0,
        max_oak_stall_s=0.02,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.exit_code != 0
    assert summary.reason == "oak_publication_stall"
    assert summary.ticks == 3
    assert source.calls == 4
    assert estimator.update_calls == 3


def test_run_soak_checks_repeated_publication_before_duration_completion():
    module = _load_module()
    clock = ManualClock()
    source = FakeSource([_sample(7, 0.0, 0.08)])

    summary = module.run_soak(
        source=source,
        estimator=FakeEstimator([_pressure("baseline", active=False)]),
        logger=FakeLogger(),
        duration_s=0.51,
        min_cycles=0,
        max_oak_stall_s=0.5,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.51,
    )

    assert summary.exit_code != 0
    assert summary.reason == "oak_publication_stall"
    assert summary.ticks == 1
    assert source.calls == 2


def test_run_soak_rejects_increasing_frame_with_excess_observed_timestamp_gap():
    module = _load_module()
    clock = ManualClock()
    source = FakeSource(
        [
            _sample(0, 0.0, 0.08),
            _sample(1, 0.021, 0.08),
        ]
    )
    estimator = FakeEstimator([_pressure("baseline", active=False)])
    logger = FakeLogger()

    summary = module.run_soak(
        source=source,
        estimator=estimator,
        logger=logger,
        duration_s=1.0,
        min_cycles=0,
        max_oak_stall_s=0.02,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.exit_code != 0
    assert summary.reason == "oak_observed_timestamp_gap"
    assert summary.ticks == 1
    assert estimator.update_calls == 1


def test_run_soak_exits_immediately_on_oak_failed():
    module = _load_module()
    source = FakeSource([_no_publication()], oak_failed=True)
    estimator = FakeEstimator([])
    logger = FakeLogger()

    summary = module.run_soak(
        source=source,
        estimator=estimator,
        logger=logger,
        duration_s=1.0,
        min_cycles=0,
        clock=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert summary.exit_code != 0
    assert summary.reason == "oak_failed"
    assert summary.ticks == 0
    assert source.calls == 0
    assert logger.rows == []


def test_run_soak_fresh_hand_loss_is_hold_and_not_an_oak_stall():
    module = _load_module()
    clock = ManualClock()
    samples = [
        _sample(
            index,
            index * 0.01,
            0.0,
            landmarks_valid=False,
            wrist_valid=False,
        )
        for index in range(4)
    ]
    estimator = FakeEstimator([])
    logger = FakeLogger()

    summary = module.run_soak(
        source=FakeSource(samples),
        estimator=estimator,
        logger=logger,
        duration_s=0.035,
        min_cycles=0,
        max_oak_stall_s=0.015,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.exit_code == 0
    assert summary.reason == "completed"
    assert summary.ticks == 4
    assert summary.cycles == 0
    assert summary.state_counts == {"HOLD": 4}
    assert summary.status_counts == {"hold": 4}
    assert estimator.update_calls == 0
    assert estimator.reset_calls == 4
    assert all(row.state == "HOLD" for row, _ in logger.rows)


def test_run_soak_flir_unavailable_continues_and_fault_latch_never_closes_more():
    module = _load_module()
    clock = ManualClock()
    logger = FakeLogger()

    summary = module.run_soak(
        source=FakeSource(
            [
                _sample(0, 0.00, 0.08),
                _sample(1, 0.01, 0.04),
                _sample(2, 0.02, 0.03),
            ]
        ),
        estimator=FakeEstimator(
            [
                _pressure("baseline", active=False),
                _pressure("thermal_unavailable", active=False, available=False),
                _pressure("thermal_unavailable", active=False, available=False),
            ]
        ),
        logger=logger,
        duration_s=0.025,
        min_cycles=0,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.exit_code != 0
    assert summary.reason == "pressure_fault_latched"
    assert summary.status_counts == {"baseline": 1, "thermal_unavailable": 2}
    assert summary.rejection_counts == {"thermal_unavailable": 2}
    assert summary.fault_closure_violations == 0
    rows = [row for row, _ in logger.rows]
    assert [row.proposed_gripper_pos for row in rows] == pytest.approx(
        [52.0, 52.0, 52.0]
    )
    assert [row.fault_latched for row in rows] == [False, True, True]
    assert [row.fallback_reason for row in rows] == [None, "thermal_unavailable", "thermal_unavailable"]


def test_run_soak_transient_flir_fault_can_rearm_on_baseline_and_complete():
    module = _load_module()
    clock = ManualClock()
    logger = FakeLogger()

    summary = module.run_soak(
        source=FakeSource(
            [
                _sample(0, 0.00, 0.08),
                _sample(1, 0.01, 0.04),
                _sample(2, 0.02, 0.08),
            ]
        ),
        estimator=FakeEstimator(
            [
                _pressure("baseline", active=False),
                _pressure("thermal_unavailable", active=False, available=False),
                _pressure("baseline", active=False),
            ]
        ),
        logger=logger,
        duration_s=0.025,
        min_cycles=0,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.exit_code == 0
    assert summary.reason == "completed"
    assert summary.status_counts == {"baseline": 2, "thermal_unavailable": 1}
    assert summary.rejection_counts == {"thermal_unavailable": 1}
    rows = [row for row, _ in logger.rows]
    assert [row.fault_latched for row in rows] == [False, True, False]
    assert [row.baseline_ready for row in rows] == [True, False, True]


def test_run_soak_fails_immediately_when_sidecar_disables_after_finalize():
    module = _load_module()
    clock = ManualClock()
    logger = FakeLogger(disable_after=1)

    summary = module.run_soak(
        source=FakeSource([_sample(0, 0.0, 0.08)]),
        estimator=FakeEstimator([_pressure("baseline", active=False)]),
        logger=logger,
        duration_s=1.0,
        min_cycles=0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert summary.exit_code != 0
    assert summary.reason == "sidecar_disabled"
    assert summary.ticks == 1
    assert len(logger.rows) == 1
    assert logger.rows[0][1] is False


def test_run_soak_checks_disabled_sidecar_at_zero_duration_completion():
    module = _load_module()
    logger = FakeLogger(enabled=False)
    source = FakeSource([_sample(0, 0.0, 0.08)])

    summary = module.run_soak(
        source=source,
        estimator=FakeEstimator([]),
        logger=logger,
        duration_s=0.0,
        min_cycles=0,
        clock=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert summary.exit_code != 0
    assert summary.reason == "sidecar_disabled"
    assert summary.ticks == 0
    assert source.calls == 0


def test_run_soak_returns_explicit_error_when_sidecar_finalize_raises():
    module = _load_module()
    logger = FakeLogger(finalize_error=OSError("disk full"))

    summary = module.run_soak(
        source=FakeSource([_sample(0, 0.0, 0.08)]),
        estimator=FakeEstimator([_pressure("baseline", active=False)]),
        logger=logger,
        duration_s=1.0,
        min_cycles=0,
        clock=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert summary.exit_code != 0
    assert summary.reason == "sidecar_error:OSError"
    assert summary.ticks == 0
    assert logger.rows == []


def test_run_soak_returns_explicit_error_when_source_read_raises():
    module = _load_module()

    summary = module.run_soak(
        source=FakeSource([_no_publication()], error=RuntimeError("source broke")),
        estimator=FakeEstimator([]),
        logger=FakeLogger(),
        duration_s=1.0,
        min_cycles=0,
        clock=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert summary.exit_code != 0
    assert summary.reason == "source_error:RuntimeError"
    assert summary.ticks == 0


@pytest.mark.parametrize(
    ("sample", "estimator"),
    [
        (
            _sample(0, 0.0, 0.08),
            FakeEstimator([], update_error=RuntimeError("update broke")),
        ),
        (
            _sample(0, 0.0, 0.0, wrist_valid=False),
            FakeEstimator([], reset_error=RuntimeError("reset broke")),
        ),
    ],
)
def test_run_soak_returns_explicit_error_when_estimator_raises(sample, estimator):
    module = _load_module()
    logger = FakeLogger()

    summary = module.run_soak(
        source=FakeSource([sample]),
        estimator=estimator,
        logger=logger,
        duration_s=1.0,
        min_cycles=0,
        clock=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert summary.exit_code != 0
    assert summary.reason == "estimator_error:RuntimeError"
    assert summary.ticks == 0
    assert logger.rows == []


def test_run_soak_fault_closure_violation_overrides_cycle_success(monkeypatch):
    module = _load_module()

    class ClosingFaultProposal:
        def __init__(self, **_kwargs):
            self.index = 0

        def update(self, base_gripper, _pressure_reading):
            proposed = [50.0, 49.0, 51.0][self.index]
            fault_latched = self.index == 1
            self.index += 1
            return SimpleNamespace(
                base_gripper=base_gripper,
                proposed_gripper=proposed,
                state="fault_latched" if fault_latched else "armed",
                fault_latched=fault_latched,
                reason="thermal_unavailable" if fault_latched else "baseline",
            )

    monkeypatch.setattr(module, "PressureProposalStateMachine", ClosingFaultProposal)
    clock = ManualClock()
    logger = FakeLogger()

    summary = module.run_soak(
        source=FakeSource(
            [
                _sample(0, 0.00, 0.08),
                _sample(1, 0.01, 0.04),
                _sample(2, 0.02, 0.08),
            ]
        ),
        estimator=FakeEstimator(
            [
                _pressure("baseline", active=False),
                _pressure("active", active=True),
                _pressure("baseline", active=False),
            ]
        ),
        logger=logger,
        duration_s=0.025,
        min_cycles=1,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.cycles == 1
    assert summary.fault_closure_violations == 1
    assert summary.exit_code != 0
    assert summary.reason == "fault_closure_violation"


def test_run_soak_reports_percentiles_for_all_required_metrics():
    module = _load_module()
    clock = ManualClock()
    observed_at_s = [0.0, 0.011, 0.023, 0.036]
    pressures = [
        _pressure(
            "baseline",
            active=False,
            oak_age_s=value / 1000.0,
            thermal_age_s=(value + 4.0) / 1000.0,
            sensor_skew_s=(value + 8.0) / 1000.0,
        )
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    logger = FakeLogger(
        clock=clock,
        finalize_delays=(0.001, 0.002, 0.003, 0.004),
    )

    summary = module.run_soak(
        source=FakeSource(
            [
                _sample(index, observed, 0.08)
                for index, observed in enumerate(observed_at_s)
            ]
        ),
        estimator=FakeEstimator(pressures),
        logger=logger,
        duration_s=0.039,
        min_cycles=0,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.exit_code == 0
    assert summary.ticks == 4
    _assert_metric(
        summary.metrics["oak_age_ms"],
        count=4,
        p50=2.5,
        p95=3.85,
        p99=3.97,
        maximum=4.0,
    )
    _assert_metric(
        summary.metrics["thermal_age_ms"],
        count=4,
        p50=6.5,
        p95=7.85,
        p99=7.97,
        maximum=8.0,
    )
    _assert_metric(
        summary.metrics["pair_skew_ms"],
        count=4,
        p50=10.5,
        p95=11.85,
        p99=11.97,
        maximum=12.0,
    )
    _assert_metric(
        summary.metrics["loop_period_ms"],
        count=3,
        p50=12.0,
        p95=12.9,
        p99=12.98,
        maximum=13.0,
    )
    _assert_metric(
        summary.metrics["control_latency_ms"],
        count=4,
        p50=2.5,
        p95=3.85,
        p99=3.97,
        maximum=4.0,
    )


def test_run_soak_reports_explicit_empty_metrics_without_observations():
    module = _load_module()

    summary = module.run_soak(
        source=FakeSource([_no_publication()]),
        estimator=FakeEstimator([]),
        logger=FakeLogger(),
        duration_s=0.0,
        min_cycles=0,
        clock=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert summary.exit_code == 0
    assert summary.reason == "completed"
    assert summary.metrics == {
        name: {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
        for name in (
            "oak_age_ms",
            "thermal_age_ms",
            "pair_skew_ms",
            "loop_period_ms",
            "control_latency_ms",
        )
    }


def test_run_soak_insufficient_cycles_is_an_explicit_nonzero_exit():
    module = _load_module()
    clock = ManualClock()

    summary = module.run_soak(
        source=FakeSource([_sample(0, 0.0, 0.08)]),
        estimator=FakeEstimator([_pressure("baseline", active=False)]),
        logger=FakeLogger(),
        duration_s=0.005,
        min_cycles=1,
        clock=clock,
        sleep=clock.sleep,
    )

    assert summary.exit_code != 0
    assert summary.reason == "insufficient_cycles"
    assert summary.ticks == 1
    assert summary.cycles == 0


def test_run_soak_progress_callback_is_deterministic_and_cannot_change_result():
    module = _load_module()
    clock = ManualClock()
    reports = []

    def failing_progress(report):
        reports.append(report)
        raise RuntimeError("reporting must not affect the soak")

    summary = module.run_soak(
        source=FakeSource(
            [_sample(index, index * 0.01, 0.08) for index in range(4)]
        ),
        estimator=FakeEstimator(
            [_pressure("baseline", active=False) for _ in range(4)]
        ),
        logger=FakeLogger(),
        duration_s=0.035,
        min_cycles=0,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
        progress=failing_progress,
        progress_interval_s=0.02,
    )

    assert summary.exit_code == 0
    assert summary.ticks == 4
    assert reports == [
        {
            "elapsed_s": pytest.approx(0.02),
            "ticks": 3,
            "cycles": 0,
            "state": "MOVING",
            "status": "baseline",
        }
    ]


def _runtime_calibration():
    return SimpleNamespace(
        coeff_x=(80.0, 0.0, 0.0, 0.0),
        coeff_y=(64.0, 0.0, 0.0, 0.0),
        rms_error_px=0.0,
        max_error_px=0.0,
        sample_count=12,
        image_size=(160, 128),
    )


def _thermal_frame(index):
    base = np.tile(np.arange(160, dtype=np.uint8), (128, 1))
    return np.stack((base, np.roll(base, index, axis=1), base), axis=2)


class RealTimeOakSource:
    oak_failed = False

    def __init__(self):
        self.frame_id = 0

    def latest_sample(self):
        observed_at_s = time.perf_counter()
        self.frame_id += 1
        sample = _sample(self.frame_id, observed_at_s, 0.08)
        sample.landmarks.image_xy = np.full((21, 2), 0.5)
        sample.landmarks.depth_m = np.full(21, 0.5)
        return sample


class CadencedThermalSource:
    def __init__(self, cadence_s=0.08, *, recover_after_s=None):
        self.cadence_s = cadence_s
        self.recover_after_s = recover_after_s
        self.read_calls = 0
        self.closed = threading.Event()

    def read(self):
        self.read_calls += 1
        if self.read_calls == 1:
            delay_s = 0.0
        elif self.read_calls == 2 and self.recover_after_s is not None:
            delay_s = self.recover_after_s
        elif self.recover_after_s is not None:
            self.closed.wait()
            raise RuntimeError("thermal source closed")
        else:
            delay_s = self.cadence_s
        if delay_s:
            time.sleep(delay_s)
        return SimpleNamespace(
            t=time.perf_counter(),
            frame=_thermal_frame(self.read_calls),
        )

    def close(self):
        self.closed.set()


class FirstThenBlockedThermalSource(CadencedThermalSource):
    def __init__(self):
        super().__init__(recover_after_s=60.0)

    def read(self):
        self.read_calls += 1
        if self.read_calls == 1:
            return SimpleNamespace(t=time.perf_counter(), frame=_thermal_frame(1))
        self.closed.wait()
        raise RuntimeError("thermal source closed")


def _wait_for_latest(latest, timeout_s=1.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            return latest.read()
        except RuntimeError:
            time.sleep(0.001)
    raise AssertionError("thermal producer did not publish before timeout")


def _real_runtime_estimator(module, latest, *, max_thermal_age_s=0.2):
    estimator = module.HandPressureEstimator(
        calibration=_runtime_calibration(),
        thermal_source=latest,
        config=module.PressureConfig(
            max_oak_age_s=0.2,
            max_thermal_age_s=max_thermal_age_s,
            max_pair_skew_s=0.2,
        ),
    )
    return module.PublicationGatedPressureEstimator(estimator, latest)


def test_real_slower_flir_cadence_has_no_false_stale_or_final_latch():
    module = _load_module()
    latest = module.LatestFrameSource(CadencedThermalSource(cadence_s=0.08))
    _wait_for_latest(latest)
    estimator = _real_runtime_estimator(module, latest)
    logger = FakeLogger()
    try:
        summary = module.run_soak(
            source=RealTimeOakSource(),
            estimator=estimator,
            logger=logger,
            duration_s=0.28,
            min_cycles=0,
            poll_interval_s=0.01,
            max_oak_stall_s=0.5,
        )
    finally:
        estimator.close()

    assert summary.exit_code == 0
    assert summary.reason == "completed"
    assert 3 <= summary.status_counts.get("baseline", 0) <= 5
    assert "thermal_stale" not in summary.status_counts
    assert summary.rejection_counts == {}
    assert all(command_sent is False for _, command_sent in logger.rows)


def test_publication_gated_estimator_processes_each_generation_or_error_once():
    module = _load_module()
    clock = ManualClock()

    class PublicationSource:
        def __init__(self):
            self.state = module.ThermalPublicationClaim(0, None, None, 0.0, None)

        def claim_publication(self):
            return self.state

    class CountingEstimator:
        config = SimpleNamespace(max_frame_age_s=None, max_thermal_age_s=1.0)

        def __init__(self):
            self.update_calls = 0
            self.reset_calls = 0
            self.close_calls = 0

        def update(self, landmarks, pinch, enabled):
            self.update_calls += 1
            return _pressure("baseline", active=False)

        def reset(self):
            self.reset_calls += 1

        def close(self):
            self.close_calls += 1

    source = PublicationSource()
    wrapped = CountingEstimator()
    estimator = module.PublicationGatedPressureEstimator(
        wrapped,
        source,
        clock=clock,
    )
    landmarks = _sample(1, 0.0, 0.08).landmarks

    source.state = module.ThermalPublicationClaim(1, 0.0, None, 0.0, None)
    first = estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)
    repeats = [
        estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)
        for _ in range(3)
    ]
    source.state = module.ThermalPublicationClaim(2, 0.0, None, 0.0, None)
    second = estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)
    source.state = module.ThermalPublicationClaim(
        3,
        0.0,
        RuntimeError("read failed"),
        0.0,
        None,
    )
    error = estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)
    repeated_error = estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)

    assert first[0] is True
    assert repeats == [(False, None)] * 3
    assert second[0] is True
    assert error[0] is True
    assert repeated_error == (False, None)
    assert wrapped.update_calls == 3


def _manual_latest_source(module, sample):
    latest = module.LatestFrameSource.__new__(module.LatestFrameSource)
    latest._lock = threading.Lock()
    latest._latest = sample
    latest._error = None
    latest._generation = 1
    latest._source_started_at_s = sample.t
    latest._running = True
    return latest


def test_atomic_claim_processes_racing_generations_once_without_false_stale():
    module = _load_module()
    now_s = time.perf_counter()
    first_sample = SimpleNamespace(t=now_s - 0.02, frame=_thermal_frame(1))
    second_sample = SimpleNamespace(t=now_s - 0.01, frame=_thermal_frame(2))
    latest = _manual_latest_source(module, first_sample)
    inner = module.HandPressureEstimator(
        calibration=_runtime_calibration(),
        thermal_source=latest,
        config=module.PressureConfig(
            max_oak_age_s=1.0,
            max_thermal_age_s=1.0,
            max_pair_skew_s=1.0,
        ),
    )
    real_update = inner.update
    update_calls = 0

    def publish_during_first_update(landmarks, pinch, enabled):
        nonlocal update_calls
        update_calls += 1
        if update_calls == 1:
            with latest._lock:
                latest._latest = second_sample
                latest._generation = 2
        return real_update(landmarks, pinch, enabled)

    inner.update = publish_during_first_update
    estimator = module.PublicationGatedPressureEstimator(inner, latest)
    landmarks = _sample(1, now_s, 0.08).landmarks
    landmarks.image_xy = np.full((21, 2), 0.5)
    landmarks.depth_m = np.full(21, 0.5)

    first = estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)
    second = estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)
    repeat = estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)

    assert first[0] is True
    assert second[0] is True
    assert repeat == (False, None)
    assert [first[1].thermal_observed_at_s, second[1].thermal_observed_at_s] == [
        first_sample.t,
        second_sample.t,
    ]
    assert [first[1].status, second[1].status] == ["baseline", "baseline"]


def test_atomic_claim_preserves_exact_error_across_racing_publication():
    module = _load_module()
    first_sample = SimpleNamespace(t=0.0, frame=_thermal_frame(1))
    latest = _manual_latest_source(module, first_sample)
    expected_error = RuntimeError("exact producer error")

    class ErrorCapturingEstimator:
        config = SimpleNamespace(max_frame_age_s=None, max_thermal_age_s=1.0)

        def __init__(self):
            self.thermal_source = latest
            self.read_results = []

        def update(self, landmarks, pinch, enabled):
            if not self.read_results:
                with latest._lock:
                    latest._error = expected_error
                    latest._generation = 2
            try:
                self.read_results.append(self.thermal_source.read())
                return _pressure("baseline", active=False)
            except Exception as exc:
                self.read_results.append(exc)
                return _pressure(
                    "thermal_unavailable",
                    active=False,
                    available=False,
                )

        def reset(self):
            pass

        def close(self):
            pass

    inner = ErrorCapturingEstimator()
    estimator = module.PublicationGatedPressureEstimator(inner, latest)
    landmarks = _sample(1, 0.0, 0.08).landmarks

    first = estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)
    error = estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)
    repeat = estimator.update_if_ready(landmarks, pinch=0.08, enabled=True)

    assert first[1].status == "baseline"
    assert error[1].status == "thermal_unavailable"
    assert repeat == (False, None)
    assert inner.read_results == [first_sample, expected_error]


class HealthSequenceEstimator:
    config = SimpleNamespace(max_frame_age_s=None, max_thermal_age_s=1.0)

    def __init__(self, thermal_source):
        self.thermal_source = thermal_source
        self.update_calls = 0
        self.reset_calls = 0

    def update(self, landmarks, pinch, enabled):
        self.thermal_source.read()
        self.update_calls += 1
        return _pressure("baseline", active=False)

    def reset(self):
        self.reset_calls += 1

    def close(self):
        pass


class ThermalPublishingOakSource:
    oak_failed = False

    def __init__(self, samples, thermal_events, latest, clock):
        self.samples = list(samples)
        self.thermal_events = list(thermal_events)
        self.latest = latest
        self.clock = clock
        self.index = 0

    def latest_sample(self):
        index = min(self.index, len(self.samples) - 1)
        sample = self.samples[index]
        if self.index < len(self.thermal_events):
            event = self.thermal_events[self.index]
            with self.latest._lock:
                self.latest._generation += 1
                if isinstance(event, Exception):
                    self.latest._error = event
                else:
                    self.latest._latest = SimpleNamespace(
                        t=self.clock(),
                        frame=_thermal_frame(self.latest._generation),
                    )
                    self.latest._error = None
        self.index += 1
        return sample


def _run_health_sequence(
    module,
    samples,
    thermal_events,
    *,
    duration_s,
    min_cycles=0,
):
    clock = ManualClock()
    seed = SimpleNamespace(t=0.0, frame=_thermal_frame(0))
    latest = _manual_latest_source(module, seed)
    latest._latest = None
    latest._generation = 0
    inner = HealthSequenceEstimator(latest)
    estimator = module.PublicationGatedPressureEstimator(
        inner,
        latest,
        clock=clock,
    )
    logger = FakeLogger()
    summary = module.run_soak(
        source=ThermalPublishingOakSource(
            samples,
            thermal_events,
            latest,
            clock,
        ),
        estimator=estimator,
        logger=logger,
        duration_s=duration_s,
        min_cycles=min_cycles,
        max_oak_stall_s=0.5,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )
    return summary, logger, inner


def _completed_cycle_then_terminal_sample(*, terminal_state):
    return [
        _sample(1, 0.0, 0.08),
        _sample(2, 0.01, 0.03),
        _sample(3, 0.02, 0.08),
        _sample(
            4,
            0.03,
            0.08,
            landmarks_valid=terminal_state != "HOLD",
            wrist_valid=terminal_state != "HOLD",
            fist_state="closed" if terminal_state == "MIDDLE" else "open",
        ),
    ]


@pytest.mark.parametrize("terminal_state", ["HOLD", "MIDDLE"])
def test_hold_or_middle_pending_thermal_error_latches_after_completed_cycle(
    terminal_state,
):
    module = _load_module()
    summary, logger, _inner = _run_health_sequence(
        module,
        _completed_cycle_then_terminal_sample(terminal_state=terminal_state),
        ["healthy", "healthy", "healthy", RuntimeError("flir stopped")],
        duration_s=0.06,
        min_cycles=1,
    )

    assert summary.cycles == 1
    assert summary.exit_code != 0
    assert summary.reason == "pressure_fault_latched"
    assert summary.fault_closure_violations == 0
    fault_row, command_sent = logger.rows[3]
    assert fault_row.state == terminal_state
    assert fault_row.pressure_status == "thermal_unavailable"
    assert fault_row.fault_latched is True
    assert command_sent is False
    assert fault_row.proposed_gripper_pos >= logger.rows[2][0].proposed_gripper_pos
    assert [row.pressure_status for row, _ in logger.rows[3:]] == [
        "thermal_unavailable",
        "fault_latched",
        "fault_latched",
    ]
    assert all(row.fault_latched for row, _ in logger.rows[3:])


def test_pending_thermal_error_at_duration_boundary_beats_success():
    module = _load_module()
    summary, logger, _inner = _run_health_sequence(
        module,
        _completed_cycle_then_terminal_sample(terminal_state="HOLD"),
        ["healthy", "healthy", "healthy", RuntimeError("boundary error")],
        duration_s=0.03,
        min_cycles=1,
    )

    assert summary.exit_code != 0
    assert summary.reason == "pressure_fault_latched"
    assert logger.rows[-1][0].pressure_status == "thermal_unavailable"
    assert logger.rows[-1][0].fault_latched is True


def test_healthy_hold_middle_publications_are_acked_without_pressure_updates():
    module = _load_module()
    samples = [
        _sample(1, 0.0, 0.08, landmarks_valid=False, wrist_valid=False),
        _sample(2, 0.01, 0.08, fist_state="closed"),
        _sample(3, 0.02, 0.08),
    ]
    summary, logger, inner = _run_health_sequence(
        module,
        samples,
        ["healthy", "healthy", "healthy"],
        duration_s=0.03,
    )

    assert summary.exit_code == 0
    assert [row.pressure_status for row, _ in logger.rows] == [
        "hold",
        "middle",
        "baseline",
    ]
    assert inner.update_calls == 1
    assert all(not row.fault_latched for row, _ in logger.rows)


def test_moving_baseline_is_required_to_rearm_after_hold_thermal_fault():
    module = _load_module()
    samples = [
        _sample(1, 0.0, 0.08),
        _sample(2, 0.01, 0.08, landmarks_valid=False, wrist_valid=False),
        _sample(3, 0.02, 0.08, landmarks_valid=False, wrist_valid=False),
        _sample(4, 0.03, 0.08),
    ]
    events = ["healthy", RuntimeError("flir stopped"), "healthy", "healthy"]

    held_summary, _held_logger, _held_inner = _run_health_sequence(
        module,
        samples[:3],
        events[:3],
        duration_s=0.02,
    )
    recovered_summary, logger, _inner = _run_health_sequence(
        module,
        samples,
        events,
        duration_s=0.04,
    )

    assert held_summary.reason == "pressure_fault_latched"
    assert recovered_summary.exit_code == 0
    assert recovered_summary.reason == "completed"
    assert [row.pressure_status for row, _ in logger.rows] == [
        "baseline",
        "thermal_unavailable",
        "fault_latched",
        "baseline",
    ]
    assert [row.fault_latched for row, _ in logger.rows] == [
        False,
        True,
        True,
        False,
    ]


def test_blocked_thermal_producer_reports_one_stale_fault_after_age_limit():
    module = _load_module()
    latest = module.LatestFrameSource(FirstThenBlockedThermalSource())
    _wait_for_latest(latest)
    estimator = _real_runtime_estimator(module, latest, max_thermal_age_s=0.04)
    try:
        summary = module.run_soak(
            source=RealTimeOakSource(),
            estimator=estimator,
            logger=FakeLogger(),
            duration_s=0.09,
            min_cycles=0,
            poll_interval_s=0.005,
            max_oak_stall_s=0.5,
        )
    finally:
        estimator.close()

    assert summary.exit_code != 0
    assert summary.reason == "pressure_fault_latched"
    assert summary.status_counts == {"baseline": 1, "thermal_stale": 1}
    assert summary.rejection_counts == {"thermal_stale": 1}


def test_oak_watchdog_remains_active_while_thermal_has_no_new_publication():
    module = _load_module()
    clock = ManualClock()

    class WaitingEstimator:
        def __init__(self):
            self.calls = 0

        def update_if_ready(self, landmarks, pinch, enabled):
            self.calls += 1
            return False, None

        def reset(self):
            pass

    estimator = WaitingEstimator()
    summary = module.run_soak(
        source=FakeSource([_sample(7, 0.0, 0.08)]),
        estimator=estimator,
        logger=FakeLogger(),
        duration_s=1.0,
        min_cycles=0,
        max_oak_stall_s=0.02,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_s=0.01,
    )

    assert summary.exit_code != 0
    assert summary.reason == "oak_publication_stall"
    assert summary.ticks == 0
    assert estimator.calls == 3


def test_new_thermal_publication_recovers_stale_latch_with_baseline_rearm():
    module = _load_module()
    latest = module.LatestFrameSource(
        CadencedThermalSource(recover_after_s=0.08)
    )
    _wait_for_latest(latest)
    estimator = _real_runtime_estimator(module, latest, max_thermal_age_s=0.04)
    logger = FakeLogger()
    try:
        summary = module.run_soak(
            source=RealTimeOakSource(),
            estimator=estimator,
            logger=logger,
            duration_s=0.11,
            min_cycles=0,
            poll_interval_s=0.005,
            max_oak_stall_s=0.5,
        )
    finally:
        estimator.close()

    assert summary.exit_code == 0
    assert summary.reason == "completed"
    assert summary.status_counts == {"baseline": 2, "thermal_stale": 1}
    assert summary.rejection_counts == {"thermal_stale": 1}
    rows = [row for row, _ in logger.rows]
    assert [row.fault_latched for row in rows] == [False, True, False]
    assert [row.baseline_ready for row in rows] == [True, False, True]


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "+inf", "-inf"])
def test_cli_rejects_nonfinite_or_negative_duration(value):
    module = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--sidecar", "soak.csv", "--duration-s", value])


@pytest.mark.parametrize("value", ["-1", "0", "nan", "inf", "+inf", "-inf"])
def test_cli_rejects_nonfinite_or_nonpositive_watchdog(value):
    module = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--sidecar", "soak.csv", "--max-oak-stall-ms", value])


class TrackedCleanupResource:
    def __init__(self, name, events, *, child=None, cleanup_error=False):
        self.name = name
        self.events = events
        self.child = child
        self.cleanup_error = cleanup_error
        self.cleanup_calls = 0

    def _cleanup(self, method):
        self.cleanup_calls += 1
        self.events.append(f"{self.name}.{method}")
        if self.child is not None:
            self.child.close()
        if self.cleanup_error:
            raise RuntimeError(f"{self.name} cleanup failed")

    def close(self):
        self._cleanup("close")


class TrackedWebcamSource(TrackedCleanupResource):
    def __init__(self, events, *, start_error=False, cleanup_error=False):
        super().__init__("oak_source", events, cleanup_error=cleanup_error)
        self.start_error = start_error
        self.start_calls = 0

    def start_oak(self):
        self.start_calls += 1
        self.events.append("oak_source.start_oak")
        if self.start_error:
            raise RuntimeError("start_oak failed")

    def stop(self):
        self._cleanup("stop")


class TrackedLogger(TrackedCleanupResource):
    def __init__(self, events, *, enabled=True, cleanup_error=False):
        super().__init__("sidecar_logger", events, cleanup_error=cleanup_error)
        self.enabled = enabled
        self.path = Path("/tmp/gate1.csv")


class RuntimeFactoryHarness:
    STAGES = (
        "load_calibration",
        "validate_calibration",
        "sidecar_logger",
        "scale_depth",
        "wrist_estimator",
        "webcam_source",
        "thermal_source",
        "latest_frame_source",
        "pressure_estimator",
    )

    def __init__(
        self,
        *,
        fail_stage=None,
        logger_enabled=True,
        cleanup_errors=(),
        latest_supports_claim=False,
    ):
        self.fail_stage = fail_stage
        self.logger_enabled = logger_enabled
        self.cleanup_errors = set(cleanup_errors)
        self.latest_supports_claim = latest_supports_claim
        self.events = []
        self.validation_kwargs = None
        self.calibration = object()
        self.depth = object()
        self.wrist = object()
        self.source = None
        self.thermal = None
        self.latest = None
        self.estimator = None
        self.logger = None

    def _enter(self, stage):
        self.events.append(stage)
        if self.fail_stage == stage:
            raise RuntimeError(f"{stage} failed")

    def load_calibration(self, path):
        self._enter("load_calibration")
        assert path == Path("calibration.json")
        return self.calibration

    def validate_calibration(self, calibration, **kwargs):
        assert calibration is self.calibration
        self.validation_kwargs = kwargs
        self._enter("validate_calibration")
        return calibration

    def make_logger(self, path):
        self._enter("sidecar_logger")
        assert path == "sidecar.csv"
        self.logger = TrackedLogger(
            self.events,
            enabled=self.logger_enabled,
            cleanup_error="sidecar_logger" in self.cleanup_errors,
        )
        return self.logger

    def make_depth(self):
        self._enter("scale_depth")
        return self.depth

    def make_wrist(self, depth):
        self._enter("wrist_estimator")
        assert depth is self.depth
        return self.wrist

    def make_webcam(self, wrist):
        self._enter("webcam_source")
        assert wrist is self.wrist
        self.source = TrackedWebcamSource(
            self.events,
            start_error=self.fail_stage == "start_oak",
            cleanup_error="oak_source" in self.cleanup_errors,
        )
        return self.source

    def make_thermal(self, path):
        self._enter("thermal_source")
        assert path == "/dev/fake-thermal"
        self.thermal = TrackedCleanupResource(
            "thermal_source",
            self.events,
            cleanup_error="thermal_source" in self.cleanup_errors,
        )
        return self.thermal

    def make_latest(self, thermal):
        self._enter("latest_frame_source")
        assert thermal is self.thermal
        self.latest = TrackedCleanupResource(
            "latest_frame_source",
            self.events,
            child=thermal,
            cleanup_error="latest_frame_source" in self.cleanup_errors,
        )
        if self.latest_supports_claim:
            self.latest.claim_publication = lambda: None
        return self.latest

    def make_estimator(self, *, calibration, thermal_source):
        self._enter("pressure_estimator")
        assert calibration is self.calibration
        assert thermal_source is self.latest
        self.estimator = TrackedCleanupResource(
            "pressure_estimator",
            self.events,
            child=thermal_source,
            cleanup_error="pressure_estimator" in self.cleanup_errors,
        )
        return self.estimator

    def factories(self, module):
        return module.RuntimeFactories(
            load_calibration=self.load_calibration,
            validate_calibration=self.validate_calibration,
            logger_factory=self.make_logger,
            scale_depth_factory=self.make_depth,
            wrist_estimator_factory=self.make_wrist,
            webcam_source_factory=self.make_webcam,
            thermal_source_factory=self.make_thermal,
            latest_frame_source_factory=self.make_latest,
            pressure_estimator_factory=self.make_estimator,
        )


def _runtime_args():
    return SimpleNamespace(
        calibration="calibration.json",
        sidecar="sidecar.csv",
        thermal="/dev/fake-thermal",
        duration_s=12.0,
        min_cycles=3,
        max_oak_stall_ms=500.0,
    )


def _runtime_summary(module, *, exit_code=0, reason="completed"):
    return module.SoakSummary(
        exit_code=exit_code,
        reason=reason,
        ticks=7,
        cycles=2,
        state_counts={"MOVING": 7},
        status_counts={"baseline": 6, "thermal_unavailable": 1},
        rejection_counts={"thermal_unavailable": 1},
        fault_closure_violations=0,
        sidecar="sidecar.csv",
        metrics={
            name: {"count": 1, "p50": 1.0, "p95": 2.0, "p99": 3.0, "max": 4.0}
            for name in (
                "oak_age_ms",
                "thermal_age_ms",
                "pair_skew_ms",
                "loop_period_ms",
                "control_latency_ms",
            )
        },
    )


def test_build_runtime_validates_calibration_first_with_gate1_thresholds():
    module = _load_module()
    harness = RuntimeFactoryHarness(fail_stage="validate_calibration")

    summary = module.execute_soak(
        _runtime_args(),
        factories=harness.factories(module),
        soak_runner=lambda **_kwargs: pytest.fail("run_soak must not start"),
    )

    assert summary.exit_code != 0
    assert summary.reason == "setup_error:validate_calibration:RuntimeError"
    assert harness.events == ["load_calibration", "validate_calibration"]
    assert harness.validation_kwargs == {
        "min_samples": 12,
        "max_rms_error_px": 8.0,
        "max_error_px": 16.0,
        "expected_image_size": (160, 128),
    }


def test_build_runtime_constructs_real_shape_and_uses_exactly_once_ownership_cleanup():
    module = _load_module()
    harness = RuntimeFactoryHarness()

    runtime = module.build_runtime(_runtime_args(), factories=harness.factories(module))

    assert runtime.source is harness.source
    assert runtime.estimator is harness.estimator
    assert runtime.logger is harness.logger
    assert harness.events == [
        "load_calibration",
        "validate_calibration",
        "sidecar_logger",
        "scale_depth",
        "wrist_estimator",
        "webcam_source",
        "oak_source.start_oak",
        "thermal_source",
        "latest_frame_source",
        "pressure_estimator",
    ]

    assert runtime.cleanup() == ()
    assert runtime.cleanup() == ()
    assert harness.events[-5:] == [
        "pressure_estimator.close",
        "latest_frame_source.close",
        "thermal_source.close",
        "oak_source.stop",
        "sidecar_logger.close",
    ]
    for resource in (
        harness.estimator,
        harness.latest,
        harness.thermal,
        harness.source,
        harness.logger,
    ):
        assert resource.cleanup_calls == 1


def test_logger_disabled_is_setup_failure_before_camera_construction():
    module = _load_module()
    harness = RuntimeFactoryHarness(logger_enabled=False)

    summary = module.execute_soak(
        _runtime_args(),
        factories=harness.factories(module),
        soak_runner=lambda **_kwargs: pytest.fail("run_soak must not start"),
    )

    assert summary.exit_code != 0
    assert summary.reason == "sidecar_disabled"
    assert harness.events == [
        "load_calibration",
        "validate_calibration",
        "sidecar_logger",
        "sidecar_logger.close",
    ]
    assert harness.logger.cleanup_calls == 1


@pytest.mark.parametrize(
    ("fail_stage", "expected_cleanup"),
    [
        ("load_calibration", []),
        ("sidecar_logger", []),
        ("scale_depth", ["sidecar_logger.close"]),
        ("wrist_estimator", ["sidecar_logger.close"]),
        ("webcam_source", ["sidecar_logger.close"]),
        ("start_oak", ["oak_source.stop", "sidecar_logger.close"]),
        ("thermal_source", ["oak_source.stop", "sidecar_logger.close"]),
        (
            "latest_frame_source",
            ["thermal_source.close", "oak_source.stop", "sidecar_logger.close"],
        ),
        (
            "pressure_estimator",
            [
                "latest_frame_source.close",
                "thermal_source.close",
                "oak_source.stop",
                "sidecar_logger.close",
            ],
        ),
    ],
)
def test_each_partial_setup_failure_cleans_only_successful_top_level_owners(
    fail_stage, expected_cleanup
):
    module = _load_module()
    harness = RuntimeFactoryHarness(fail_stage=fail_stage)

    summary = module.execute_soak(
        _runtime_args(),
        factories=harness.factories(module),
        soak_runner=lambda **_kwargs: pytest.fail("run_soak must not start"),
    )

    assert summary.exit_code != 0
    assert summary.reason.startswith("setup_error:")
    cleanup_events = [event for event in harness.events if ".close" in event or ".stop" in event]
    assert cleanup_events == expected_cleanup


def test_publication_wrapper_setup_failure_keeps_inner_estimator_ownership():
    module = _load_module()
    harness = RuntimeFactoryHarness(latest_supports_claim=True)

    summary = module.execute_soak(
        _runtime_args(),
        factories=harness.factories(module),
        soak_runner=lambda **_kwargs: pytest.fail("run_soak must not start"),
    )

    assert summary.exit_code != 0
    assert summary.reason == "setup_error:publication_pressure_estimator:AttributeError"
    assert harness.estimator.cleanup_calls == 1
    assert harness.latest.cleanup_calls == 1
    assert harness.source.cleanup_calls == 1
    assert harness.logger.cleanup_calls == 1
    assert harness.events[-5:] == [
        "pressure_estimator.close",
        "latest_frame_source.close",
        "thermal_source.close",
        "oak_source.stop",
        "sidecar_logger.close",
    ]


def test_initial_flir_open_failure_is_setup_failure_and_never_runs_core():
    module = _load_module()
    harness = RuntimeFactoryHarness(fail_stage="thermal_source")

    summary = module.execute_soak(
        _runtime_args(),
        factories=harness.factories(module),
        soak_runner=lambda **_kwargs: pytest.fail("run_soak must not start"),
    )

    assert summary.reason == "setup_error:thermal_source:RuntimeError"
    assert harness.source.cleanup_calls == 1
    assert harness.logger.cleanup_calls == 1


def test_cleanup_attempts_all_owners_and_makes_success_nonzero():
    module = _load_module()
    harness = RuntimeFactoryHarness(
        cleanup_errors={"pressure_estimator", "oak_source"}
    )

    summary = module.execute_soak(
        _runtime_args(),
        factories=harness.factories(module),
        soak_runner=lambda **_kwargs: _runtime_summary(module),
    )

    assert summary.exit_code != 0
    assert summary.reason == "cleanup_failed"
    assert summary.cleanup_failures == (
        "pressure_estimator.close:RuntimeError:pressure_estimator cleanup failed",
        "oak_source.stop:RuntimeError:oak_source cleanup failed",
    )
    assert harness.logger.cleanup_calls == 1
    assert harness.events[-5:] == [
        "pressure_estimator.close",
        "latest_frame_source.close",
        "thermal_source.close",
        "oak_source.stop",
        "sidecar_logger.close",
    ]


def test_cleanup_failure_preserves_existing_specific_nonzero_result():
    module = _load_module()
    harness = RuntimeFactoryHarness(cleanup_errors={"sidecar_logger"})

    summary = module.execute_soak(
        _runtime_args(),
        factories=harness.factories(module),
        soak_runner=lambda **_kwargs: _runtime_summary(
            module, exit_code=10, reason="oak_failed"
        ),
    )

    assert summary.exit_code == 10
    assert summary.reason == "oak_failed"
    assert summary.cleanup_failures == (
        "sidecar_logger.close:RuntimeError:sidecar_logger cleanup failed",
    )


@pytest.mark.parametrize(
    ("runner_result", "expected_code", "expected_reason"),
    [
        ("normal", 0, "completed"),
        ("nonzero", 10, "oak_failed"),
        ("keyboard", 130, "keyboard_interrupt"),
        ("exception", 22, "runtime_error:RuntimeError"),
    ],
)
def test_main_handles_normal_nonzero_interrupt_and_runtime_exception(
    runner_result, expected_code, expected_reason
):
    module = _load_module()
    harness = RuntimeFactoryHarness()
    output = []

    def runner(**kwargs):
        harness.events.append("run_soak")
        assert kwargs["max_oak_stall_s"] == pytest.approx(0.5)
        assert kwargs["duration_s"] == 12.0
        assert kwargs["min_cycles"] == 3
        kwargs["progress"](
            {
                "elapsed_s": 10.0,
                "ticks": 5,
                "cycles": 1,
                "state": "MOVING",
                "status": "baseline",
            }
        )
        if runner_result == "keyboard":
            raise KeyboardInterrupt
        if runner_result == "exception":
            raise RuntimeError("unexpected")
        if runner_result == "nonzero":
            return _runtime_summary(module, exit_code=10, reason="oak_failed")
        return _runtime_summary(module)

    exit_code = module.main(
        [
            "--sidecar",
            "sidecar.csv",
            "--calibration",
            "calibration.json",
            "--thermal",
            "/dev/fake-thermal",
            "--duration-s",
            "12",
            "--min-cycles",
            "3",
            "--max-oak-stall-ms",
            "500",
        ],
        factories=harness.factories(module),
        soak_runner=runner,
        output=output.append,
    )

    assert exit_code == expected_code
    assert harness.events[-5:] == [
        "pressure_estimator.close",
        "latest_frame_source.close",
        "thermal_source.close",
        "oak_source.stop",
        "sidecar_logger.close",
    ]
    assert len(output) == 2
    progress = json.loads(output[0])
    final = json.loads(output[1])
    assert progress == {
        "cycles": 1,
        "elapsed_s": 10.0,
        "ticks": 5,
        "state": "MOVING",
        "status": "baseline",
        "type": "progress",
    }
    assert final["type"] == "summary"
    assert final["exit_code"] == expected_code
    assert final["reason"] == expected_reason
    assert final["cleanup_failures"] == []
    assert set(final["metrics"]) == {
        "oak_age_ms",
        "thermal_age_ms",
        "pair_skew_ms",
        "loop_period_ms",
        "control_latency_ms",
    }
    expected_rejections = (
        {"thermal_unavailable": 1}
        if runner_result in {"normal", "nonzero"}
        else {}
    )
    assert final["rejection_counts"] == expected_rejections


def test_main_setup_exception_prints_final_summary_without_entering_run_soak():
    module = _load_module()
    harness = RuntimeFactoryHarness(fail_stage="load_calibration")
    output = []

    exit_code = module.main(
        ["--sidecar", "sidecar.csv", "--calibration", "calibration.json"],
        factories=harness.factories(module),
        soak_runner=lambda **_kwargs: pytest.fail("run_soak must not start"),
        output=output.append,
    )

    assert exit_code != 0
    assert len(output) == 1
    final = json.loads(output[0])
    assert final["type"] == "summary"
    assert final["reason"] == "setup_error:load_calibration:RuntimeError"
    assert final["sidecar"] == "sidecar.csv"


def test_runtime_system_exit_propagates_after_cleanup():
    module = _load_module()
    harness = RuntimeFactoryHarness()

    def exit_runner(**_kwargs):
        raise SystemExit(7)

    with pytest.raises(SystemExit) as exc_info:
        module.main(
            [
                "--sidecar",
                "sidecar.csv",
                "--calibration",
                "calibration.json",
                "--thermal",
                "/dev/fake-thermal",
            ],
            factories=harness.factories(module),
            soak_runner=exit_runner,
            output=lambda _line: None,
        )

    assert exc_info.value.code == 7
    assert harness.source.cleanup_calls == 1
    assert harness.estimator.cleanup_calls == 1
    assert harness.logger.cleanup_calls == 1


def test_fresh_setup_failure_executes_without_robot_modules_or_traceback():
    result = _fresh_python(
        f"""
import json
import sys
import ir_pressure_soak

exit_code = ir_pressure_soak.main([
    '--sidecar', '/tmp/unused-gate1.csv',
    '--calibration', '/definitely/missing/gate1-calibration.json',
    '--duration-s', '0',
    '--min-cycles', '0',
])
prohibited = sorted(
    name for name in sys.modules
    if name == 'serial'
    or name == 'ir_force.ir_robot'
    or name.startswith('serial.')
    or name.startswith('lerobot.teleoperators')
    or name.startswith('lerobot.robots')
    or name.startswith('lerobot.motors')
    or '.motors.' in name
    or name.endswith('teleop_viz_ee')
    or name.endswith('record_so101_ee')
    or '.bus' in name.lower()
    or '.action' in name.lower()
    or '.observation' in name.lower()
)
print(json.dumps({{
    'exit_code': exit_code,
    'prohibited': prohibited,
    'soak_modules': sorted(name for name in sys.modules
                       if name.startswith('lerobot') or name.startswith('ir_force')),
}}))
"""
    )
    lines = result.stdout.splitlines()
    final = json.loads(lines[-2])
    payload = json.loads(lines[-1])

    assert final["type"] == "summary"
    assert final["reason"] == "setup_error:load_calibration:FileNotFoundError"
    assert payload["exit_code"] != 0
    assert payload["prohibited"] == []
    assert payload["soak_modules"] == ROBOT_FREE_MODULES
    assert result.stderr == ""


def test_main_is_not_a_placeholder():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "runtime wiring is not implemented" not in source
    assert "Task R5 Phase B2c is required" not in source
