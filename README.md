**TinyML-Based Environmental Sound Classification for Hearing-Impaired Users**
This project presents an efficient TinyML solution for real-time environmental sound classification aimed at assisting hearing-impaired individuals. The system recognizes critical sounds like sirens, car horns, and gunshots, and displays the results on an OLED screen using an embedded microcontroller.

**Overview**
Deploys a compact and accurate CNN model on ESP32

Uses the UrbanSound8K dataset with 10 environmental sound classes

Processes audio through log-Mel spectrograms

Delivers real-time classification with minimal latency and low memory footprint

**Key Features**
Model: Optimized ShuffleNetV2 with knowledge distillation

Accuracy: 98.74% on test data

Model Size: 154.2 KB (quantized)

Inference Time: ~0.71 ms

Hardware:

ESP32

INMP441 I2S digital MEMS microphone

SSD1306 128x64 OLED display via I2C

**Audio Classes**
Air Conditioner

Car Horn

Children Playing

Dog Bark

Drilling

Engine Idling

Gun Shot

Jackhammer

Siren

Street Music

**Dataset and Preprocessing**
Dataset: UrbanSound8K

Feature Extraction: 128x62 log-Mel spectrograms

Data Augmentation:

Time stretching

Pitch shifting

Time shifting

Gain adjustment

Class-specific effects (e.g., stutter for car horn, high-pass filtering for gun shot)

**Workflow**
Audio Capture: Real-time I2S input from INMP441

Preprocessing: On-device generation of log-Mel spectrograms

Inference: On-device classification using ShuffleNetV2

Output: Predicted label shown on SSD1306 OLED display

**Deployment**
The final model is deployed on ESP32 using Eloquent Tinyml Library in Arduino IDE. Quantized ShuffleNetV2 provides a balanced trade-off between accuracy, latency, and memory for real-world deployment.

**Future Work**
Add sound direction estimation

Support more sound classes

Deploy on wearable devices for continuous monitoring
