#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H

#include "student_int8_new.h"                // TFLite model
#include <tflm_esp32.h>                      // TensorFlow Lite Micro
#include <eloquent_tinyml.h>                 // EloquentTinyML wrapper

// Model configuration
#define N_INPUTS    (128 * 62)               // Input size (128 mel bands x 62 time steps)
#define N_OUTPUTS   10                       // Number of classes
#define TF_NUM_OPS  200                      // Number of ops used in the model
#define ARENA_SIZE  (104 * 1024)             // PSRAM arena size

// Quantization parameters
const float INPUT_QUANT_SCALE = 0.3137255012989044f;
const int INPUT_QUANT_ZERO_POINT = 127;
const float OUTPUT_QUANT_SCALE = 0.00390625f;
const int OUTPUT_QUANT_ZERO_POINT = -128;

namespace TinyMLRunner {
    using namespace Eloquent::TF;

    Sequential<TF_NUM_OPS, ARENA_SIZE> tf;
    bool model_initialized = false;

    bool init_model() {
        if (model_initialized) return true;

        tf.setNumInputs(N_INPUTS);
        tf.setNumOutputs(N_OUTPUTS);

        tf.resolver.AddConv2D();
        tf.resolver.AddDepthwiseConv2D();
        tf.resolver.AddReshape();
        tf.resolver.AddTranspose();
        tf.resolver.AddConcatenation();
        tf.resolver.AddAveragePool2D();
        tf.resolver.AddAdd();
        tf.resolver.AddRelu();
        tf.resolver.AddQuantize();
        tf.resolver.AddDequantize();
        tf.resolver.AddFullyConnected();
        tf.resolver.AddSoftmax();
        tf.resolver.AddStridedSlice();
        tf.resolver.AddShape();
        tf.resolver.AddMaxPool2D();
        tf.resolver.AddMean();
        tf.resolver.AddPack();

        uint8_t* arena = (uint8_t*)ps_malloc(ARENA_SIZE);
        if (!arena) {
            Serial.println("[ERROR] Failed to allocate tensor arena in PSRAM");
            return false;
        }
        tf.setArena(arena, ARENA_SIZE);

        if (!tf.begin(student_int8_new).isOk()) {
            Serial.println(tf.exception.toString());
            return false;
        }

        model_initialized = true;
        return true;
    }

    /**
     * Run inference on float32 input
     * 
     * @param input  Pointer to float[7936] buffer (128x62)
     * @param output Float array (length 10) for output probabilities
     * @return true if inference succeeded
     */
    bool run_model(int8_t* input, float* output) {
        Serial.println("========== [RUN MODEL] ==========");

        if (!init_model()) {
            Serial.println("[ERROR] Model initialization failed");
            return false;
        }

        if (input == nullptr) {
            Serial.println("[ERROR] Input pointer is NULL!");
            return false;
        }


        Serial.println("[DEBUG] Sample of float32 spectrogram input (first 32 values):");
        for (int i = 0; i < 32; i++) {
            Serial.print(input[i]);
            Serial.print(i < 31 ? ", " : "\n");
        }

        Serial.println("[DEBUG] Running inference...");
        if (!tf.predict(input).isOk()) {
            Serial.print("[ERROR] Inference failed: ");
            Serial.println(tf.exception.toString());
            return false;
        }
        Serial.println("[DEBUG] Inference successful.");

        Serial.println("[DEBUG] Model outputs (dequantized):");
        for (int i = 0; i < N_OUTPUTS; i++) {
            output[i] = tf.output(i);  // Already dequantized
            Serial.printf("  [%d] = %.4f\n", i, output[i]);
        }

        Serial.println("========== [/RUN MODEL] ==========");
        return true;
    }

} // namespace TinyMLRunner

#endif // MODEL_RUNNER_H
