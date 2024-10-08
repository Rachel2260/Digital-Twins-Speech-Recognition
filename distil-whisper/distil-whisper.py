import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import speech_recognition as sr
import queue
import threading
import time
import numpy as np
import noisereduce as nr

# Define device and model parameters
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-medium.en"

# Load model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

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
                print(transcription, flush=True)
            else:
                print("Error: 'input_features' not found in inputs")

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
