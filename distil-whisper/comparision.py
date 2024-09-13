import torch
import time
import torchaudio
import psutil
import noisereduce as nr
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, AutoModelForSpeechSeq2Seq
from jiwer import cer
import os

# Define device and model parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper and Distil-Whisper models
whisper_model_id = "openai/whisper-medium.en"
distil_model_id = "distil-whisper/distil-medium.en"

whisper_processor = WhisperProcessor.from_pretrained(whisper_model_id)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_id).to(device)

distil_processor = AutoProcessor.from_pretrained(distil_model_id)
distil_model = AutoModelForSpeechSeq2Seq.from_pretrained(distil_model_id).to(device)

# Function to measure GPU memory usage
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    return 0

# Function to measure CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

# Function to perform transcription and measure performance
def transcribe_and_measure(model, processor, audio_data, device, torch_dtype):
    # Noise reduction
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=16000)
    
    # Process the audio block
    inputs = processor(reduced_noise_audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    input_features = inputs['input_features'].to(device)
    
    # Measure start time and memory usage
    start_time = time.time()
    start_gpu_memory = get_gpu_memory_usage()
    start_cpu_usage = get_cpu_usage()

    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Measure end time and memory usage
    end_time = time.time()
    end_gpu_memory = get_gpu_memory_usage()
    end_cpu_usage = get_cpu_usage()

    # Calculate time, GPU, and CPU usage
    total_time = end_time - start_time
    gpu_memory_used = end_gpu_memory - start_gpu_memory
    cpu_usage = end_cpu_usage - start_cpu_usage

    return transcription, total_time, gpu_memory_used, cpu_usage

# Compare both models
def compare_models(whisper_model, distil_model, whisper_processor, distil_processor, audio_data, true_transcription):
    print("Starting comparison...\n")

    # Transcribe using Whisper
    print("Testing Whisper Model:")
    whisper_transcription, whisper_time, whisper_gpu, whisper_cpu = transcribe_and_measure(
        whisper_model, whisper_processor, audio_data, device, torch_dtype
    )
    whisper_cer = cer(true_transcription, whisper_transcription)
    print(f"Whisper Transcription: {whisper_transcription}")
    print(f"Whisper CER: {whisper_cer:.4f}, Time: {whisper_time:.2f}s, GPU Memory: {whisper_gpu:.2f}MB, CPU Usage: {whisper_cpu:.2f}%\n")

    # Transcribe using Distil-Whisper
    print("Testing Distil-Whisper Model:")
    distil_transcription, distil_time, distil_gpu, distil_cpu = transcribe_and_measure(
        distil_model, distil_processor, audio_data, device, torch_dtype
    )
    distil_cer = cer(true_transcription, distil_transcription)
    print(f"Distil-Whisper Transcription: {distil_transcription}")
    print(f"Distil-Whisper CER: {distil_cer:.4f}, Time: {distil_time:.2f}s, GPU Memory: {distil_gpu:.2f}MB, CPU Usage: {distil_cpu:.2f}%\n")

    # Compare CER
    print(f"Character Error Rate (CER) Comparison: Whisper vs Distil-Whisper = {whisper_cer:.4f} vs {distil_cer:.4f}")

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

# Compare the models on the loaded audio sample
compare_models(whisper_model, distil_model, whisper_processor, distil_processor, audio_data.squeeze(), true_transcription)
