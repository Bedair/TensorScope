#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr std::size_t kTensorArenaSize = 2 * 1024 * 1024;
constexpr std::size_t kArenaAlignment = 16;
constexpr int kResolverCapacity = 8;

alignas(kArenaAlignment)
std::uint8_t tensor_arena[kTensorArenaSize];

void PrintUsage(const char* program_name) {
  std::fprintf(
      stderr,
      "Usage: %s MODEL.tflite\n",
      program_name);
}

bool ReadBinaryFile(
    const char* path,
    std::vector<std::uint8_t>* output) {
  if (output == nullptr) {
    return false;
  }

  std::ifstream input(
      path,
      std::ios::binary);

  if (!input.is_open()) {
    std::fprintf(
        stderr,
        "Error: failed to open model file: %s\n",
        path);

    return false;
  }

  input.unsetf(std::ios::skipws);

  output->assign(
      std::istream_iterator<std::uint8_t>(input),
      std::istream_iterator<std::uint8_t>());

  if (input.bad()) {
    std::fprintf(
        stderr,
        "Error: failed while reading model file: %s\n",
        path);

    return false;
  }

  if (output->empty()) {
    std::fprintf(
        stderr,
        "Error: model file is empty: %s\n",
        path);

    return false;
  }

  return true;
}

void ResetTensorArena() {
  for (std::size_t index = 0;
       index < kTensorArenaSize;
       ++index) {
    tensor_arena[index] = 0;
  }
}

TfLiteStatus RegisterCorpusOperators(
    tflite::MicroMutableOpResolver<
        kResolverCapacity>* resolver) {
  if (resolver == nullptr) {
    return kTfLiteError;
  }

  if (resolver->AddAdd() != kTfLiteOk) {
    std::fprintf(
        stderr,
        "Error: failed to register ADD.\n");

    return kTfLiteError;
  }

  if (resolver->AddConv2D() != kTfLiteOk) {
    std::fprintf(
        stderr,
        "Error: failed to register CONV_2D.\n");

    return kTfLiteError;
  }

  if (resolver->AddDepthwiseConv2D() != kTfLiteOk) {
    std::fprintf(
        stderr,
        "Error: failed to register DEPTHWISE_CONV_2D.\n");

    return kTfLiteError;
  }

  if (resolver->AddFullyConnected() != kTfLiteOk) {
    std::fprintf(
        stderr,
        "Error: failed to register FULLY_CONNECTED.\n");

    return kTfLiteError;
  }

  if (resolver->AddReshape() != kTfLiteOk) {
    std::fprintf(
        stderr,
        "Error: failed to register RESHAPE.\n");

    return kTfLiteError;
  }

  if (resolver->AddSoftmax() != kTfLiteOk) {
    std::fprintf(
        stderr,
        "Error: failed to register SOFTMAX.\n");

    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

int main(
    int argc,
    char** argv) {
  if (argc != 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  const char* model_path = argv[1];

  std::vector<std::uint8_t> model_data;

  if (!ReadBinaryFile(
          model_path,
          &model_data)) {
    return 2;
  }

  if (model_data.size() < 8) {
    std::fprintf(
        stderr,
        "Error: model file is too small: %s\n",
        model_path);

    return 3;
  }

  const tflite::Model* model =
      tflite::GetModel(model_data.data());

  if (model == nullptr) {
    std::fprintf(
        stderr,
        "Error: failed to parse model: %s\n",
        model_path);

    return 4;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::fprintf(
        stderr,
        "Error: model schema version %d does not match "
        "runtime schema version %d.\n",
        model->version(),
        TFLITE_SCHEMA_VERSION);

    return 5;
  }

  ResetTensorArena();

  tflite::MicroMutableOpResolver<
      kResolverCapacity> resolver;

  if (RegisterCorpusOperators(
          &resolver) != kTfLiteOk) {
    return 6;
  }

  tflite::RecordingMicroInterpreter interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::fprintf(
        stderr,
        "Error: tensor allocation failed for model: %s\n",
        model_path);

    return 7;
  }

  const auto* subgraphs = model->subgraphs();
  const auto* operator_codes = model->operator_codes();

  const unsigned int subgraph_count =
      subgraphs == nullptr
          ? 0
          : static_cast<unsigned int>(
                subgraphs->size());

  const unsigned int operator_code_count =
      operator_codes == nullptr
          ? 0
          : static_cast<unsigned int>(
                operator_codes->size());

  std::printf("TENSOR_SCOPE_ORACLE_BEGIN\n");
  std::printf("model_path=%s\n", model_path);
  std::printf(
      "model_size=%zu\n",
      model_data.size());
  std::printf(
      "schema_version=%d\n",
      model->version());
  std::printf(
      "subgraph_count=%u\n",
      subgraph_count);
  std::printf(
      "operator_code_count=%u\n",
      operator_code_count);
  std::printf(
      "arena_capacity=%zu\n",
      kTensorArenaSize);
  std::printf(
      "arena_used=%zu\n",
      interpreter.arena_used_bytes());
  std::printf("TENSOR_SCOPE_ORACLE_END\n");

  interpreter
      .GetMicroAllocator()
      .PrintAllocations();

  return 0;
}