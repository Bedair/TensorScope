TFLM_ROOT := ../../third_party/tflite-micro
TFLM_MAKEFILE := tensorflow/lite/micro/tools/make/Makefile

GEN_ROOT := $(TFLM_ROOT)/gen/linux_x86_64_default_gcc
GENFILES := $(GEN_ROOT)/genfiles
TFLM_LIB := $(GEN_ROOT)/lib/libtensorflow-microlite.a

TARGET := build/tflm_oracle

CXX := g++

CXXFLAGS := \
	-std=c++17 \
	-O2 \
	-fno-rtti \
	-fno-exceptions \
	-fno-threadsafe-statics \
	-DTF_LITE_STATIC_MEMORY \
	-DTF_LITE_DISABLE_X86_NEON \
	-DTF_LITE_USE_CTIME

INCLUDES := \
	-I$(TFLM_ROOT) \
	-I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads \
	-I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
	-I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
	-I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/kissfft \
	-I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/ruy \
	-I$(GENFILES)

.PHONY: all run-corpus clean tflm

all: $(TARGET)

tflm:
	$(MAKE) -C $(TFLM_ROOT) \
		-f $(TFLM_MAKEFILE) \
		-j$$(nproc) \
		microlite

$(TARGET): tflm main.cc
	mkdir -p build
	$(CXX) \
		$(CXXFLAGS) \
		$(INCLUDES) \
		main.cc \
		$(TFLM_LIB) \
		-lm \
		-o $(TARGET)

run-corpus: $(TARGET)
	./$(TARGET) ../../tests/model_corpus/models/hello_world_float.tflite
	./$(TARGET) ../../tests/model_corpus/models/hello_world_int8.tflite
	./$(TARGET) ../../tests/model_corpus/models/simple_add_model.tflite
	./$(TARGET) ../../tests/model_corpus/models/conv0.tflite
	./$(TARGET) ../../tests/model_corpus/models/micro_speech_quantized.tflite

clean:
	rm -rf build