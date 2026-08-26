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
constexpr int kResolverCapacity = 19;
constexpr const char* kTflmRevision =
    "b89fb3e06e59d2f6af67e758242243da599bfedf";

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

#define REGISTER_TARGET(method, name)                                      \
  if (resolver->method() != kTfLiteOk) {                                   \
    std::fprintf(stderr, "Error: failed to register " name ".\n");       \
    return kTfLiteError;                                                    \
  }
  REGISTER_TARGET(AddMul, "MUL")
  REGISTER_TARGET(AddMaxPool2D, "MAX_POOL_2D")
  REGISTER_TARGET(AddAveragePool2D, "AVERAGE_POOL_2D")
  REGISTER_TARGET(AddRelu, "RELU")
  REGISTER_TARGET(AddRelu6, "RELU6")
  REGISTER_TARGET(AddLogistic, "LOGISTIC")
  REGISTER_TARGET(AddQuantize, "QUANTIZE")
  REGISTER_TARGET(AddDequantize, "DEQUANTIZE")
  REGISTER_TARGET(AddPad, "PAD")
  REGISTER_TARGET(AddStridedSlice, "STRIDED_SLICE")
  REGISTER_TARGET(AddSub, "SUB")
  REGISTER_TARGET(AddTransposeConv, "TRANSPOSE_CONV")
  REGISTER_TARGET(AddLeakyRelu, "LEAKY_RELU")
#undef REGISTER_TARGET

  return kTfLiteOk;
}

bool IsRegisteredBuiltin(tflite::BuiltinOperator code) {
  switch (code) {
    case tflite::BuiltinOperator_ADD: case tflite::BuiltinOperator_MUL:
    case tflite::BuiltinOperator_RESHAPE: case tflite::BuiltinOperator_SOFTMAX:
    case tflite::BuiltinOperator_CONV_2D: case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
    case tflite::BuiltinOperator_MAX_POOL_2D: case tflite::BuiltinOperator_AVERAGE_POOL_2D:
    case tflite::BuiltinOperator_FULLY_CONNECTED: case tflite::BuiltinOperator_RELU:
    case tflite::BuiltinOperator_RELU6: case tflite::BuiltinOperator_LOGISTIC:
    case tflite::BuiltinOperator_QUANTIZE: case tflite::BuiltinOperator_DEQUANTIZE:
    case tflite::BuiltinOperator_PAD: case tflite::BuiltinOperator_STRIDED_SLICE:
    case tflite::BuiltinOperator_SUB: case tflite::BuiltinOperator_TRANSPOSE_CONV:
    case tflite::BuiltinOperator_LEAKY_RELU:
      return true;
    default: return false;
  }
}

bool ValidateRegisteredOperators(const tflite::Model* model, const char* model_path) {
  const auto* codes = model->operator_codes();
  const auto* subgraphs = model->subgraphs();
  if (codes == nullptr || subgraphs == nullptr) return false;
  for (unsigned int sg = 0; sg < subgraphs->size(); ++sg) {
    const auto* operators = subgraphs->Get(sg)->operators();
    if (operators == nullptr) continue;
    for (unsigned int index = 0; index < operators->size(); ++index) {
      const auto* op = operators->Get(index);
      if (op->opcode_index() >= codes->size()) continue;
      const auto* code = codes->Get(op->opcode_index());
      const auto builtin = code->builtin_code();
      if (!IsRegisteredBuiltin(builtin)) {
        const char* custom = code->custom_code() == nullptr ? "" : code->custom_code()->c_str();
        std::fprintf(stderr,
                     "Error: unsupported operator at index %u in model %s: builtin opcode %d, custom code '%s'.\n",
                     index, model_path, static_cast<int>(builtin), custom);
        return false;
      }
    }
  }
  return true;
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

  if (!ValidateRegisteredOperators(model, model_path)) {
    return 6;
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

  const auto* arena_allocator = interpreter
      .GetMicroAllocator()
      .GetSimpleMemoryAllocator();
  const std::size_t arena_head_bytes =
      arena_allocator->GetNonPersistentUsedBytes();
  const std::size_t arena_tail_bytes =
      arena_allocator->GetPersistentUsedBytes();
  const std::size_t arena_used_bytes = arena_allocator->GetUsedBytes();
  const std::size_t arena_remaining_bytes =
      kTensorArenaSize - arena_used_bytes;

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
      arena_used_bytes);
  std::printf("arena_head_bytes=%zu\n", arena_head_bytes);
  std::printf("arena_tail_bytes=%zu\n", arena_tail_bytes);
  std::printf("arena_temporary_bytes=unavailable\n");
  std::printf("arena_remaining_bytes=%zu\n", arena_remaining_bytes);
  std::printf("allocator_alignment_bytes=%zu\n", kArenaAlignment);
  std::printf("scratch_observation_available=false\n");
  std::printf("scratch_request_count=unavailable\n");
  std::printf("scratch_requested_total_bytes=unavailable\n");
  std::printf("scratch_peak_bytes=unavailable\n");
  std::printf("scratch_operator_attribution=unavailable\n");
  std::printf("tflm_revision=%s\n", kTflmRevision);
  std::printf("TENSOR_SCOPE_ORACLE_END\n");
  std::fflush(stdout);

  interpreter
      .GetMicroAllocator()
      .PrintAllocations();

  return 0;
}
