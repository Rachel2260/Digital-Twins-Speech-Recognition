import torch
import time
import torchaudio
import psutil
import noisereduce as nr
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, AutoModelForSpeechSeq2Seq
from jiwer import wer
import os
import numpy as np

# Define device and model parameters
device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper and Distil-Whisper models
whisper_model_id = "openai/whisper-medium.en"
distil_model_id = "distil-whisper/distil-medium.en"

whisper_processor = WhisperProcessor.from_pretrained(whisper_model_id)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_id).to(device)

distil_processor = AutoProcessor.from_pretrained(distil_model_id)
distil_model = AutoModelForSpeechSeq2Seq.from_pretrained(distil_model_id).to(device)


# Function to measure CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

# Function to pad mel features and create attention mask
def pad_mel_features(mel_features, target_length=3000):
    """Pad mel features to the target length with -1.0 and create an attention mask."""
    pad_width = target_length - mel_features.shape[-1]
    attention_mask = np.ones(mel_features.shape[-1])
    
    if pad_width > 0:
        # Padding with -1.0 as required by Whisper model
        mel_features = np.pad(mel_features, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1.0)
        # Padding the attention mask with 0 for the padded regions
        attention_mask = np.pad(attention_mask, (0, pad_width), mode='constant', constant_values=0)
        
    return mel_features, attention_mask

# Modify transcribe_and_measure function to include multiple runs and median timing
def transcribe_and_measure(model, processor, audio_data, device, torch_dtype, n_runs=5):
    """Perform inference multiple times to calculate median inference time."""
    # Noise reduction
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=16000)

    # Process the audio block to get input features
    inputs = processor(reduced_noise_audio, sampling_rate=16000, return_tensors="pt", padding=True)

    if 'input_features' in inputs:
        # Extract the input features and squeeze to remove extra dimensions
        input_features = inputs['input_features'].squeeze().cpu().numpy()

        # Pad the input features to the required length and create attention mask
        input_features, attention_mask = pad_mel_features(input_features)
        input_features = torch.tensor(input_features).unsqueeze(0).to(device, dtype=torch_dtype)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device, dtype=torch_dtype)

        # Measure inference time over multiple runs
        timers = []
        for _ in range(n_runs):
            start_time = time.time()

            # Generate transcription with attention mask
            with torch.no_grad():
                generated_ids = model.generate(input_features, attention_mask=attention_mask, max_new_tokens=128)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            end_time = time.time()
            timers.append(end_time - start_time)

        # Compute median time
        median_time = np.median(timers)

        # Measure GPU and CPU usage (just once)
        cpu_usage = get_cpu_usage()

        return transcription, median_time, cpu_usage
    else:
        raise ValueError("Error: 'input_features' not found in inputs")

# Compare both models
def compare_models(whisper_model, distil_model, whisper_processor, distil_processor, audio_data, true_transcription, n_runs=5):
    print("Starting comparison...\n")

    # Transcribe using Whisper (run multiple times and calculate median inference time)
    print("Testing Whisper Model:")
    whisper_transcription, whisper_time, whisper_cpu = transcribe_and_measure(
        whisper_model, whisper_processor, audio_data, device, torch_dtype, n_runs
    )
    whisper_wer = wer(true_transcription, whisper_transcription)
    print(f"Whisper Transcription: {whisper_transcription}")
    print(f"Whisper WER: {whisper_wer:.4f}, Median Time: {whisper_time:.2f}s, CPU Usage: {whisper_cpu:.2f}%\n")

    # Transcribe using Distil-Whisper (run multiple times and calculate median inference time)
    print("Testing Distil-Whisper Model:")
    distil_transcription, distil_time, distil_cpu = transcribe_and_measure(
        distil_model, distil_processor, audio_data, device, torch_dtype, n_runs
    )
    distil_wer = wer(true_transcription, distil_transcription)
    print(f"Distil-Whisper Transcription: {distil_transcription}")
    print(f"Distil-Whisper WER: {distil_wer:.4f}, Median Time: {distil_time:.2f}s, CPU Usage: {distil_cpu:.2f}%\n")

    # Compare wer
    print(f"Character Error Rate (wer) Comparison: Whisper vs Distil-Whisper = {whisper_wer:.4f} vs {distil_wer:.4f}")

# Load a sample from LibriSpeech dataset using torchaudio
def load_librispeech_sample():
    # Ensure the download directory exists
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)  # Create the directory if it doesn't exist

    # Download and load the "test-clean" subset of LibriSpeech
    dataset = torchaudio.datasets.LIBRISPEECH(data_dir, url="test-clean", download=True)
    
    # Get the first sample from the dataset
    waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = next(iter(dataset))

    print(f"Transcript: {transcript}")
    print(f"Waveform shape: {waveform.shape}")
    print(f"Sample rate: {sample_rate}")

    return waveform, transcript

# Load and preprocess the LibriSpeech audio sample
audio_data, true_transcription = load_librispeech_sample()

# Compare the models on the loaded audio sample using median inference time over 5 runs
compare_models(whisper_model, distil_model, whisper_processor, distil_processor, audio_data.squeeze(), true_transcription, n_runs=5)
