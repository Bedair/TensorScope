from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tensorscope.tflite.schema.schema_generated import Model


TFLITE_FILE_IDENTIFIER = b"TFL3"


class TFLiteModelError(ValueError):
    """Raised when a file is not a valid supported TFLite model."""


@dataclass(frozen=True)
class LoadedModel:
    path: Path
    data: bytes
    model: Model

    @property
    def schema_version(self) -> int:
        return int(self.model.Version())

    @property
    def subgraph_count(self) -> int:
        return int(
            self.model.SubgraphsLength()
        )

    @property
    def operator_code_count(self) -> int:
        return int(
            self.model.OperatorCodesLength()
        )

    @property
    def buffer_count(self) -> int:
        return int(
            self.model.BuffersLength()
        )


def load_tflite_model(
    model_path: str | Path,
) -> LoadedModel:
    path = Path(
        model_path
    ).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"TFLite model does not exist: {path}"
        )

    data = path.read_bytes()

    if len(data) < 8:
        raise TFLiteModelError(
            "File is too small to be a TFLite model: "
            f"{path}"
        )

    identifier = data[4:8]

    if identifier != TFLITE_FILE_IDENTIFIER:
        raise TFLiteModelError(
            "Invalid TFLite file identifier: "
            f"expected {TFLITE_FILE_IDENTIFIER!r}, "
            f"got {identifier!r}"
        )

    try:
        model = Model.GetRootAsModel(
            data,
            0,
        )
    except Exception as error:
        raise TFLiteModelError(
            "Unable to parse TFLite FlatBuffer: "
            f"{path}"
        ) from error

    if model.SubgraphsLength() == 0:
        raise TFLiteModelError(
            "TFLite model contains no subgraphs: "
            f"{path}"
        )

    return LoadedModel(
        path=path,
        data=data,
        model=model,
    )