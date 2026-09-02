"""Experiments may only call the public API the way it actually exists.

`teleop_viz_ee.py` called `WebcamEEController` with ten keyword arguments the
public controller does not accept. Import succeeded, 1015 tests passed, and the
failure would only have appeared with a robot connected -- the constructor is
inside a function body, so nothing evaluated it.

The public repo has an attribute guard for the same class of bug. It would not
have caught this one: the defect was in the call's keywords, not in an
attribute. So this checks both, against the real imported signature rather than
a copy of it.
"""

import ast
import inspect
from pathlib import Path

import pytest

from lerobot_teleoperator_so101_webcam.ee_controller import WebcamEEController
from pressurevision_integration.pv_grip_controller import PressureVisionGripRuntime

EXPERIMENTS = Path(__file__).parents[1] / "experiments"
EXPERIMENT_FILES = sorted(EXPERIMENTS.glob("*.py"))

#: The classes these experiments construct, by the name they are called under.
CONSTRUCTED = {
    "WebcamEEController": WebcamEEController,
    "PressureVisionGripRuntime": PressureVisionGripRuntime,
}
#: local variable name -> the class it is bound to
BINDINGS = {"controller": WebcamEEController, "pv": PressureVisionGripRuntime}


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _accepted_keywords(cls):
    parameters = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None  # **kwargs accepts anything; nothing to check
    return {name for name in parameters if name != "self"}


def _instance_attributes(cls):
    """Class members plus anything __init__ assigns to self."""
    names = set(dir(cls))
    for node in ast.walk(ast.parse(inspect.getsource(cls))):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Store)
        ):
            names.add(node.attr)
    return names


def _probed_names(tree):
    """Attributes the file explicitly probes with hasattr/getattr.

    Declaring an attribute optional once makes it optional for the whole file:
    the guard and the use are usually a few lines apart, in the same branch.
    """
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"hasattr", "getattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return names


def test_there_are_experiments_to_check():
    assert len(EXPERIMENT_FILES) >= 10


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=lambda p: p.name)
def test_constructor_keywords_exist_on_the_real_signature(path):
    """The defect this file exists for: ten keywords the controller rejects."""
    rejected = []
    for node in ast.walk(_tree(path)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        cls = CONSTRUCTED.get(node.func.id)
        if cls is None:
            continue
        accepted = _accepted_keywords(cls)
        if accepted is None:
            continue
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg not in accepted:
                rejected.append(
                    f"{path.name}:{node.lineno} {node.func.id}({keyword.arg}=...) "
                    f"is not a parameter of {cls.__name__}"
                )
    assert not rejected, "\n".join(rejected)


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=lambda p: p.name)
def test_experiment_uses_only_real_attributes(path):
    tree = _tree(path)
    optional = _probed_names(tree)
    dangling = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        target = node.value
        name = (
            target.id
            if isinstance(target, ast.Name)
            else target.attr
            if isinstance(target, ast.Attribute)
            else None
        )
        cls = BINDINGS.get(name)
        if cls is None or node.attr in optional:
            continue
        if node.attr not in _instance_attributes(cls):
            dangling.append(
                f"{path.name}:{node.lineno} {name}.{node.attr} not on {cls.__name__}"
            )
    assert not dangling, "\n".join(dangling)
