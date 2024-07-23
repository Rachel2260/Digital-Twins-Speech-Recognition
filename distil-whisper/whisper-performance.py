from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import speech_recognition as sr
import queue
import threading
import time
import numpy as np
import noisereduce as nr
from jiwer import wer, cer

import time

class PerformanceMetrics:
    def __init__(self):
        self.model_load_start_time = 0
        self.model_load_end_time = 0
        self.transcription_times = []
        self.total_audio_blocks = 0

    def start_model_load_timer(self):
        self.model_load_start_time = time.time()

    def end_model_load_timer(self):
        self.model_load_end_time = time.time()

    def add_transcription_time(self, start_time, end_time):
        self.transcription_times.append(end_time - start_time)
        self.total_audio_blocks += 1

    def get_model_load_time(self):
        return self.model_load_end_time - self.model_load_start_time

    def get_average_transcription_time(self):
        return sum(self.transcription_times) / len(self.transcription_times) if self.transcription_times else 0

    def get_total_transcription_time(self):
        return sum(self.transcription_times)

    def get_average_latency(self):
        return self.get_total_transcription_time() / self.total_audio_blocks if self.total_audio_blocks else 0

    def print_metrics(self):
        print(f"Model Load Time: {self.get_model_load_time():.3f} seconds")
        print(f"Average Transcription Time per Block: {self.get_average_transcription_time():.3f} seconds")
        print(f"Total Transcription Time: {self.get_total_transcription_time():.3f} seconds")
        print(f"Average Latency per Block: {self.get_average_latency():.3f} seconds")

# Define device and model parameters
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-medium.en"

# Initialize performance metrics
metrics = PerformanceMetrics()
metrics.start_model_load_timer()

# Load model and processor
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.to(device)

# End model load timer
metrics.end_model_load_timer()

# Create a queue to communicate between the audio callback and the main thread
q = queue.Queue()

def audio_callback(recognizer, audio):
    """This is called (from a separate thread) for each audio block."""
    try:
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
        q.put(audio_data)
    except Exception as e:
        print(f"Error processing audio: {e}")

def pad_mel_features(mel_features, target_length=3000):
    """Pad mel features to the target length with -1.0 and create an attention mask."""
    pad_width = target_length - mel_features.shape[-1]
    attention_mask = np.ones(mel_features.shape[-1])
    
    if pad_width > 0:
        mel_features = np.pad(mel_features, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1.0)
        attention_mask = np.pad(attention_mask, (0, pad_width), mode='constant', constant_values=0)
        
    return mel_features, attention_mask

def transcribe_audio():
    """Continuously fetch audio from the queue and run transcription."""
    while True:
        audio_block = q.get()
        if len(audio_block) > 0:
            transcription_start_time = time.time()

            # Ensure the audio block is 1D array
            audio_block = audio_block.flatten()

            # Reduce noise
            reduced_noise_audio = nr.reduce_noise(y=audio_block, sr=16000)

            # Process the audio block
            inputs = processor(reduced_noise_audio, sampling_rate=16000, return_tensors="pt", padding=True)
            # print("Inputs structure:", inputs)

            if 'input_features' in inputs:
                input_features = inputs['input_features'].squeeze().cpu().numpy()
                
                # Pad the input features to the required length and create attention mask
                input_features, attention_mask = pad_mel_features(input_features)
                input_features = torch.tensor(input_features).unsqueeze(0).to(device, dtype=torch_dtype)
                attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device, dtype=torch_dtype)

                with torch.no_grad():
                    generated_ids = model.generate(input_features, attention_mask=attention_mask, max_new_tokens=128)
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                transcription_end_time = time.time()
                metrics.add_transcription_time(transcription_start_time, transcription_end_time)
                transcriptions.append(transcription)  # Store transcription
                print(transcription, flush=True)
            else:
                print("Error: 'input_features' not found in inputs")

# Define reference text for evaluation (for demonstration purposes)
reference_text = "This is a reference transcription text for evaluation purposes. The application is designed to build the 3D models of donors' organs. The first step is to add a case for the donor. Here is the donor's basic information. A donor identified as 56088180 died on 27 October 2018 at age 74. The reason was malignant neoplasm."
transcriptions = []  # To store transcriptions

# Start a separate thread for the transcription
transcription_thread = threading.Thread(target=transcribe_audio)
transcription_thread.daemon = True
transcription_thread.start()

# Initialize the recognizer and microphone
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16000)

print("#" * 80)
print("Press Ctrl+C to stop the recording")
print("#" * 80)

# Adjust for ambient noise
with mic as source:
    recognizer.adjust_for_ambient_noise(source)

# Start recording in the background
stop_listening = recognizer.listen_in_background(mic, audio_callback)

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    stop_listening(wait_for_stop=False)
    print("\nRecording finished.")

    # Join all transcriptions for evaluation
    hypothesis_text = " ".join(transcriptions)

    # Compute WER and CER
    print("Evaluating model performance...")
    wer_score = wer(reference_text, hypothesis_text)
    cer_score = cer(reference_text, hypothesis_text)

    print(f"WER: {wer_score:.3f}")
    print(f"CER: {cer_score:.3f}")

    # Print performance metrics
    metrics.print_metrics()
