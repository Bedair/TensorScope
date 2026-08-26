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
            f"{path} ({len(data)} bytes). Check that the file "
            "downloaded, exported, or copied completely."
        )

    identifier = data[4:8]

    if identifier != TFLITE_FILE_IDENTIFIER:
        raise TFLiteModelError(
            f"{path} does not look like a valid TFLite model file. "
            "Invalid TFLite file identifier: expected "
            f"{TFLITE_FILE_IDENTIFIER!r}, got {identifier!r}. Check that "
            "this is a real .tflite file, not a different format (such as "
            "a .pb or .h5 model) or a partial/corrupted download."
        )

    try:
        model = Model.GetRootAsModel(
            data,
            0,
        )
    except Exception as error:
        raise TFLiteModelError(
            f"{path} has the right file marker but its contents could not "
            "be parsed as a TFLite model. It may be truncated, corrupted, "
            "or built with an incompatible converter version."
        ) from error

    if model.SubgraphsLength() == 0:
        raise TFLiteModelError(
            f"{path} is a valid TFLite file but contains no subgraphs, so "
            "there is nothing to analyze. It may be an empty or "
            "placeholder model."
        )

    return LoadedModel(
        path=path,
        data=data,
        model=model,
    )