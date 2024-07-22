import os
import numpy as np
import sounddevice as sd
from transformers import AutoProcessor
import openvino.runtime as ov
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from queue import Queue
from threading import Thread
from pathlib import Path

# Initialize the model and processor
quantized_model_id = "distil-whisper_distil-large-v2_quantized"
quantized_model_path = Path(quantized_model_id)

# Check if the quantized model path exists
if not quantized_model_path.exists() or not (quantized_model_path / "config.json").exists():
    raise FileNotFoundError(f"Quantized model path '{quantized_model_path}' does not exist or is missing necessary files.")

processor = AutoProcessor.from_pretrained(quantized_model_path)

ov_config = {"CACHE_DIR": ""}

# Load the quantized model from the saved path
quantized_ov_model = OVModelForSpeechSeq2Seq.from_pretrained(quantized_model_path, ov_config=ov_config, compile=False)

# Initialize OpenVINO core and device
core = ov.Core()
device_value = "AUTO"  # You can set to "CPU" if GPU is not available

# Compile and load the model on the selected device
quantized_ov_model.to(device_value)
quantized_ov_model.compile()

# Create a queue to hold audio data
audio_queue = Queue()

# Function to extract input features
def extract_input_features(audio_array, sampling_rate):
    input_features = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).input_features
    return input_features

# Audio callback function
def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())

# Function to process audio data and perform transcription
def process_audio():
    buffer = np.array([], dtype=np.float32)
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break
        
        # Accumulate audio data
        buffer = np.concatenate((buffer, audio_data[:, 0]))

        # Process buffer when it reaches a certain length (e.g., 10 seconds)
        if len(buffer) >= 16000 * 10:
            input_features = extract_input_features(buffer, 16000)
            predicted_ids = quantized_ov_model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            print(f"Transcription: {transcription[0]}", flush=True)
            buffer = np.array([], dtype=np.float32)

# Start a thread to process audio data
audio_thread = Thread(target=process_audio)
audio_thread.start()

# Start recording audio
with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
    print("Press Ctrl+C to stop the recording.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        audio_queue.put(None)

audio_thread.join()
