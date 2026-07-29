#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_float_model_data.h"

namespace {

constexpr std::size_t kTensorArenaSize = 64 * 1024;
alignas(16) std::uint8_t tensor_arena[kTensorArenaSize];

const char* TensorTypeName(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return "float32";
    case kTfLiteInt8:
      return "int8";
    case kTfLiteUInt8:
      return "uint8";
    case kTfLiteInt16:
      return "int16";
    case kTfLiteInt32:
      return "int32";
    case kTfLiteNoType:
      return "no-type";
    default:
      return "other";
  }
}

}  // namespace

int main() {
  const tflite::Model* model =
      tflite::GetModel(g_hello_world_float_model_data);

  if (model == nullptr) {
    std::fprintf(stderr, "Error: failed to load model.\n");
    return 1;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::fprintf(
        stderr,
        "Error: model schema version %d does not match runtime version %d.\n",
        model->version(),
        TFLITE_SCHEMA_VERSION);
    return 2;
  }

  tflite::MicroMutableOpResolver<1> resolver;

  if (resolver.AddFullyConnected() != kTfLiteOk) {
    std::fprintf(stderr, "Error: failed to register FullyConnected.\n");
    return 3;
  }

  tflite::MicroInterpreter interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::fprintf(stderr, "Error: tensor allocation failed.\n");
    return 4;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  if (input == nullptr || output == nullptr) {
    std::fprintf(stderr, "Error: input or output tensor unavailable.\n");
    return 5;
  }

  std::printf(
      "Input tensor type: %s (%d)\n",
      TensorTypeName(input->type),
      static_cast<int>(input->type));

  std::printf(
      "Output tensor type: %s (%d)\n",
      TensorTypeName(output->type),
      static_cast<int>(output->type));

  if (input->type != kTfLiteFloat32 ||
      output->type != kTfLiteFloat32) {
    std::fprintf(
        stderr,
        "Error: float model exposed unexpected tensor types.\n");
    return 6;
  }

  constexpr float kInputValue = 0.0f;
  input->data.f[0] = kInputValue;

  if (interpreter.Invoke() != kTfLiteOk) {
    std::fprintf(stderr, "Error: inference failed.\n");
    return 7;
  }

  std::printf("Model loaded successfully.\n");
  std::printf("Schema version: %d\n", model->version());
  std::printf("Input value: %.6f\n", kInputValue);
  std::printf("Output value: %.6f\n", output->data.f[0]);
  std::printf("Arena capacity: %zu bytes\n", kTensorArenaSize);
  std::printf(
      "Arena used: %zu bytes\n",
      interpreter.arena_used_bytes());

  return 0;
}