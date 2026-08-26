"""Reproducibly generate the tiny operator-coverage TFLite corpus models."""

from __future__ import annotations

import struct
from pathlib import Path

import flatbuffers

from tensorscope.tflite.schema import schema_generated as s


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "model_corpus" / "models"


def _buffer(data: bytes = b"") -> s.BufferT:
    return s.BufferT(data=list(data))


def _tensor(name: str, shape: list[int], buffer: int = 0) -> s.TensorT:
    return s.TensorT(shape=shape, type=s.TensorType.FLOAT32, buffer=buffer, name=name)


def _option(name: str, **values: object) -> object:
    value = getattr(s, f"{name}T")()
    for key, item in values.items():
        setattr(value, key, item)
    return value


def _write(name: str, model: s.ModelT) -> None:
    builder = flatbuffers.Builder(4096)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    (OUTPUT / name).write_bytes(builder.Output())


def generate_float_chain() -> None:
    op_names = [
        "CONV_2D", "DEPTHWISE_CONV_2D", "MAX_POOL_2D",
        "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED",
        "RELU", "RELU6", "LOGISTIC", "SOFTMAX", "MUL", "ADD",
    ]
    codes = [
        s.OperatorCodeT(deprecatedBuiltinCode=getattr(s.BuiltinOperator, name),
                        builtinCode=getattr(s.BuiltinOperator, name), version=1)
        for name in op_names
    ]
    buffers = [
        _buffer(),
        _buffer(struct.pack("<f", 1.0)),
        _buffer(struct.pack("<f", 0.0)),
        _buffer(struct.pack("<ii", 1, 1)),
        _buffer(struct.pack("<f", 1.0)),
        _buffer(struct.pack("<f", 0.0)),
        _buffer(struct.pack("<f", 0.5)),
    ]
    tensors = [
        _tensor("input", [1, 4, 4, 1]),
        _tensor("conv_weights", [1, 1, 1, 1], 1),
        _tensor("conv_bias", [1], 2),
        _tensor("conv", [1, 4, 4, 1]),
        _tensor("depthwise_weights", [1, 1, 1, 1], 1),
        _tensor("depthwise_bias", [1], 2),
        _tensor("depthwise", [1, 4, 4, 1]),
        _tensor("max_pool", [1, 2, 2, 1]),
        _tensor("average_pool", [1, 1, 1, 1]),
        s.TensorT(shape=[2], type=s.TensorType.INT32, buffer=3, name="shape"),
        _tensor("reshape", [1, 1]),
        _tensor("fc_weights", [1, 1], 4),
        _tensor("fc_bias", [1], 5),
        _tensor("fully_connected", [1, 1]),
        _tensor("relu", [1, 1]), _tensor("relu6", [1, 1]),
        _tensor("logistic", [1, 1]), _tensor("softmax", [1, 1]),
        _tensor("scalar", [1], 6), _tensor("mul", [1, 1]),
        _tensor("output", [1, 1]),
    ]
    none = s.ActivationFunctionType.NONE
    pool = dict(padding=s.Padding.VALID, strideW=2, strideH=2,
                filterWidth=2, filterHeight=2, fusedActivationFunction=none)
    specifications = [
        (0, [0, 1, 2], [3], "Conv2DOptions", dict(padding=s.Padding.SAME, strideW=1, strideH=1, dilationWFactor=1, dilationHFactor=1, fusedActivationFunction=none)),
        (1, [3, 4, 5], [6], "DepthwiseConv2DOptions", dict(padding=s.Padding.SAME, strideW=1, strideH=1, depthMultiplier=1, dilationWFactor=1, dilationHFactor=1, fusedActivationFunction=none)),
        (2, [6], [7], "Pool2DOptions", pool),
        (3, [7], [8], "Pool2DOptions", pool),
        (4, [8, 9], [10], "ReshapeOptions", dict(newShape=[1, 1])),
        (5, [10, 11, 12], [13], "FullyConnectedOptions", dict(fusedActivationFunction=none)),
        (6, [13], [14], None, {}), (7, [14], [15], None, {}),
        (8, [15], [16], None, {}),
        (9, [16], [17], "SoftmaxOptions", dict(beta=1.0)),
        (10, [17, 18], [19], "MulOptions", dict(fusedActivationFunction=none)),
        (11, [19, 17], [20], "AddOptions", dict(fusedActivationFunction=none)),
    ]
    operators = []
    for opcode, inputs, outputs, option_name, values in specifications:
        options = _option(option_name, **values) if option_name else None
        option_type = getattr(s.BuiltinOptions, option_name) if option_name else s.BuiltinOptions.NONE
        operators.append(s.OperatorT(opcodeIndex=opcode, inputs=inputs, outputs=outputs,
                                     builtinOptionsType=option_type, builtinOptions=options))
    model = s.ModelT(version=3, operatorCodes=codes,
                     subgraphs=[s.SubGraphT(tensors=tensors, inputs=[0], outputs=[20], operators=operators, name="main")],
                     description="TensorScope reproducible float operator coverage", buffers=buffers)
    _write("operator_chain_float.tflite", model)


def generate_quantize_dequantize() -> None:
    quant = s.QuantizationParametersT(scale=[0.125], zeroPoint=[-3])
    tensors = [
        _tensor("input", [1, 8]),
        s.TensorT(shape=[1, 8], type=s.TensorType.INT8, buffer=0,
                  name="quantized", quantization=quant),
        _tensor("output", [1, 8]),
    ]
    codes = [
        s.OperatorCodeT(deprecatedBuiltinCode=s.BuiltinOperator.QUANTIZE,
                        builtinCode=s.BuiltinOperator.QUANTIZE, version=1),
        s.OperatorCodeT(deprecatedBuiltinCode=s.BuiltinOperator.DEQUANTIZE,
                        builtinCode=s.BuiltinOperator.DEQUANTIZE, version=2),
    ]
    operators = [
        s.OperatorT(opcodeIndex=0, inputs=[0], outputs=[1],
                    builtinOptionsType=s.BuiltinOptions.QuantizeOptions,
                    builtinOptions=s.QuantizeOptionsT()),
        s.OperatorT(opcodeIndex=1, inputs=[1], outputs=[2],
                    builtinOptionsType=s.BuiltinOptions.DequantizeOptions,
                    builtinOptions=s.DequantizeOptionsT()),
    ]
    model = s.ModelT(version=3, operatorCodes=codes,
                     subgraphs=[s.SubGraphT(tensors=tensors, inputs=[0], outputs=[2], operators=operators, name="main")],
                     description="TensorScope reproducible quantize/dequantize coverage",
                     buffers=[_buffer()])
    _write("quantize_dequantize_int8.tflite", model)


def generate_residual_block() -> None:
    """A minimal skip connection: conv -> conv -> add-with-earlier-tensor.

    The graph input feeds op0 (CONV_2D) and is read again, unmodified, by
    op2 (ADD) after an unrelated op1 (CONV_2D) runs in between. That keeps
    the input tensor live across an operator that doesn't touch it at all,
    the same pattern documented by explain.py's ReuseBlocker: a tensor
    that survives a branch and therefore cannot hand its memory to
    anything allocated while it's still needed downstream.
    """

    codes = [
        s.OperatorCodeT(deprecatedBuiltinCode=s.BuiltinOperator.CONV_2D,
                        builtinCode=s.BuiltinOperator.CONV_2D, version=1),
        s.OperatorCodeT(deprecatedBuiltinCode=s.BuiltinOperator.ADD,
                        builtinCode=s.BuiltinOperator.ADD, version=1),
    ]
    buffers = [
        _buffer(),
        _buffer(struct.pack("<f", 1.0)),
        _buffer(struct.pack("<f", 0.0)),
        _buffer(struct.pack("<f", 1.0)),
        _buffer(struct.pack("<f", 0.0)),
    ]
    tensors = [
        _tensor("input", [1, 4, 4, 1]),
        _tensor("conv1_weights", [1, 1, 1, 1], 1),
        _tensor("conv1_bias", [1], 2),
        _tensor("conv1_out", [1, 4, 4, 1]),
        _tensor("conv2_weights", [1, 1, 1, 1], 3),
        _tensor("conv2_bias", [1], 4),
        _tensor("conv2_out", [1, 4, 4, 1]),
        _tensor("output", [1, 4, 4, 1]),
    ]
    none = s.ActivationFunctionType.NONE
    conv_options = dict(padding=s.Padding.SAME, strideW=1, strideH=1,
                        dilationWFactor=1, dilationHFactor=1, fusedActivationFunction=none)
    specifications = [
        (0, [0, 1, 2], [3], "Conv2DOptions", conv_options),
        (0, [3, 4, 5], [6], "Conv2DOptions", conv_options),
        (1, [6, 0], [7], "AddOptions", dict(fusedActivationFunction=none)),
    ]
    operators = []
    for opcode, inputs, outputs, option_name, values in specifications:
        options = _option(option_name, **values)
        option_type = getattr(s.BuiltinOptions, option_name)
        operators.append(s.OperatorT(opcodeIndex=opcode, inputs=inputs, outputs=outputs,
                                     builtinOptionsType=option_type, builtinOptions=options))
    model = s.ModelT(version=3, operatorCodes=codes,
                     subgraphs=[s.SubGraphT(tensors=tensors, inputs=[0], outputs=[7], operators=operators, name="main")],
                     description="TensorScope reproducible skip-connection reuse-blocker coverage",
                     buffers=buffers)
    _write("residual_add_float.tflite", model)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generate_float_chain()
    generate_quantize_dequantize()
    generate_residual_block()
