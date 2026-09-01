"""Test configuration.

Import paths come from `pythonpath` in pyproject.toml, which pytest applies
before collection. A `sys.path` insert here would run too late under the
`importlib` import mode this suite uses (needed because the two IR lines
contribute same-named test files).
"""
