**Environmental Sound Classification using DSP and TinyML**

This project presents an efficient TinyML solution for real-time environmental sound classification aimed at assisting hearing-impaired individuals. The system recognizes critical sounds like sirens, car horns, and gunshots, and displays the results on an OLED screen using an embedded microcontroller.

**Overview**

• Built an environmental sound classification system using ShuffleNetV2, incorporating digital signal processing (DSP) techniques such as framing, windowing, FFT, and Log-Mel spectrogram–based audio feature extraction.

• Performed knowledge distillation from a 26 MB CNN teacher model to train a 450 KB ShuffleNetV2 model using soft labels, improving its accuracy by 2% while preserving the model size.

• Applied INT8 quantization, reducing model size to 154KB while retaining 98% accuracy and deployed it on ESP-32.

**Key Features**
- **Model**: Optimized **ShuffleNetV2** with *knowledge distillation*
- **Accuracy**: 98.74% on test data
- **Model Size**: 154.2 KB (*quantized*)
- **Inference Time**: ~0.71 ms

**Hardware:**
- **ESP32 S3 N8R8**
- **INMP441** I2S digital MEMS microphone
- **SSD1306** 128x64 OLED display via I2C

**Audio Classes**
- Air Conditioner
- Car Horn
- Children Playing
- Dog Bark
- Drilling
- Engine Idling
- Gun Shot
- Jackhammer
- Siren
- Street Music

**Dataset and Preprocessing**
- **Dataset**: *UrbanSound8K*
- **Feature Extraction**: 128x62 **log-Mel spectrograms**

**Data Augmentation:**
- Time stretching
- Pitch shifting
- Time shifting
- Gain adjustment
- Class-specific effects (e.g., stutter for car horn, high-pass filtering for gun shot)

**Workflow**
- **Audio Capture**: Real-time I2S input from **INMP441**
- **Preprocessing**: On-device generation of **log-Mel spectrograms**
- **Inference**: On-device classification using **ShuffleNetV2**
- **Output**: Predicted label shown on **SSD1306** OLED display

**Deployment**
The final model is deployed on **ESP32** using the *Eloquent TinyML* library in Arduino IDE. Quantized **ShuffleNetV2** provides a balanced trade-off between **accuracy**, **latency**, and **memory** for real-world deployment.

**Future Work**
- Add *sound direction estimation*
- Support *more sound classes*
- Deploy on *wearable devices* for continuous monitoring

**Note**

ShuffleNetV2_KD.ipynb contains the Model Implementation

TinyML_Audio_Classifier folder contains Arduino Implementation
