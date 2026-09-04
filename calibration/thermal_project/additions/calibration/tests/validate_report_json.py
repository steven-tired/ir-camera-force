import json
import pathlib
import sys


report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert isinstance(report, dict)
assert report["schema_version"] == "thermal-heldout-projection/v1"
