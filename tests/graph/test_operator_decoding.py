from __future__ import annotations

from types import SimpleNamespace

from tensorscope.graph.tflite_converter import (
    _builtin_operator_name,
    _effective_builtin_code,
    _operator_name,
)
from tensorscope.tflite.schema.schema_generated import BuiltinOperator


class _Code:
    def __init__(self, builtin: int, deprecated: int, custom: bytes | None = None, version: int = 1):
        self.values = builtin, deprecated, custom, version

    def BuiltinCode(self) -> int: return self.values[0]
    def DeprecatedBuiltinCode(self) -> int: return self.values[1]
    def CustomCode(self) -> bytes | None: return self.values[2]
    def Version(self) -> int: return self.values[3]


def _loaded(code: _Code) -> SimpleNamespace:
    model = SimpleNamespace(OperatorCodes=lambda index: code)
    return SimpleNamespace(operator_code_count=1, model=model)


def test_current_builtin_code_wins_and_version_is_preserved() -> None:
    code = _Code(BuiltinOperator.MUL, BuiltinOperator.ADD, version=7)
    assert _effective_builtin_code(code) == BuiltinOperator.MUL
    assert _operator_name(_loaded(code), 0) == ("MUL", 7, BuiltinOperator.MUL, "")


def test_legacy_deprecated_builtin_code_is_used() -> None:
    code = _Code(0, BuiltinOperator.SOFTMAX, version=2)
    assert _operator_name(_loaded(code), 0) == (
        "SOFTMAX", 2, BuiltinOperator.SOFTMAX, ""
    )


def test_unknown_builtin_name_is_numeric_and_deterministic() -> None:
    assert _builtin_operator_name(999) == "BUILTIN_999"
    assert _builtin_operator_name(999) == "BUILTIN_999"


def test_custom_code_is_preserved_with_numeric_builtin() -> None:
    code = _Code(BuiltinOperator.CUSTOM, BuiltinOperator.CUSTOM, b"AcmeOp", 3)
    assert _operator_name(_loaded(code), 0) == (
        "CUSTOM:AcmeOp", 3, BuiltinOperator.CUSTOM, "AcmeOp"
    )
