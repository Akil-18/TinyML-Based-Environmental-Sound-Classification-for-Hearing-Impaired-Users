import requests
import serial
import numpy as np
import librosa
import tflite_runtime.interpreter as tflite

# CONFIGURATIONS
ESP32_URL = "http://192.168.1.123/recording.wav"  # Change to your ESP32 IP
WAV_FILE = "downloaded.wav"
MODEL_PATH = "student_int8.tflite"
SERIAL_PORT = "/dev/ttyUSB0"  # Or COM3 on Windows
BAUD_RATE = 115200
CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

# PREPROCESS PARAMETERS
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
EXPECTED_FRAMES = 62  # 1s = 62 frames with 256 hop @ 16kHz

def download_wav():
    print(f"Downloading audio from {ESP32_URL}...")
    r = requests.get(ESP32_URL)
    with open(WAV_FILE, "wb") as f:
        f.write(r.content)
    print("Download complete.")

def preprocess_log_mel(filename):
    y, sr = librosa.load(filename, sr=SAMPLE_RATE)
    if len(y) < SAMPLE_RATE:
        y = np.pad(y, (0, SAMPLE_RATE - len(y)))
    else:
        y = y[:SAMPLE_RATE]

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    if log_mel_spec.shape[1] < EXPECTED_FRAMES:
        log_mel_spec = np.pad(log_mel_spec, ((0,0), (0, EXPECTED_FRAMES - log_mel_spec.shape[1])), constant_values=-80)
    else:
        log_mel_spec = log_mel_spec[:, :EXPECTED_FRAMES]

    log_mel_spec = log_mel_spec.astype(np.float32)
    return log_mel_spec

def quantize_input(input_data, scale, zero_point):
    return np.clip(np.round(input_data / scale + zero_point), -128, 127).astype(np.int8)

def predict(log_mel_spec):
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    quantized_input = quantize_input(log_mel_spec, input_scale, input_zero_point)
    quantized_input = np.expand_dims(quantized_input, axis=0)  # Shape: (1, 128, 62)

    interpreter.set_tensor(input_details[0]['index'], quantized_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output_scale, output_zero_point = output_details[0]['quantization']
    probabilities = (output.astype(np.float32) - output_zero_point) * output_scale

    return probabilities

def send_to_arduino(prediction, probabilities):
    label = CLASS_NAMES[prediction]
    msg = f"Prediction: {label} ({probabilities[prediction]:.2f})\n"
    print("Sending to Arduino:", msg)

    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
        ser.write(msg.encode('utf-8'))

if __name__ == "__main__":
    download_wav()
    log_mel = preprocess_log_mel(WAV_FILE)
    probs = predict(log_mel)
    top = np.argmax(probs)
    send_to_arduino(top, probs)
