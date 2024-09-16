# source from https://docs.openvino.ai/2023.3/notebooks/267-distil-whisper-asr-with-output.html

from pathlib import Path
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset
import openvino as ov
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
import time
import numpy as np
from tqdm import tqdm
from itertools import islice
import gc
import psutil  # for CPU usage
from jiwer import wer, wer_standardize
from IPython.display import display
import IPython.display as ipd

# Model IDs dictionary
model_ids = {
    "Distil-Whisper": [
        "distil-whisper/distil-large-v2",
        "distil-whisper/distil-medium.en",
        "distil-whisper/distil-small.en",
    ],
    "Whisper": [
        "openai/whisper-large-v3",
        "openai/whisper-large-v2",
        "openai/whisper-large",
        "openai/whisper-medium",
        "openai/whisper-small",
        "openai/whisper-base",
        "openai/whisper-tiny",
        "openai/whisper-medium.en",
        "openai/whisper-small.en",
        "openai/whisper-base.en",
        "openai/whisper-tiny.en",
    ],
}

# Set model type and model ID programmatically
model_type_value = "Distil-Whisper"
model_id_value = model_ids[model_type_value][1]

# Initialize the processor and model using the selected model ID
processor = AutoProcessor.from_pretrained(model_id_value)
pt_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id_value)
pt_model.eval()

# Function to extract input features from a sample
def extract_input_features(sample):
    input_features = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    ).input_features
    return input_features

# Load a sample dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
sample = dataset[0]
input_features = extract_input_features(sample)

# Display the audio sample and perform transcription
predicted_ids = pt_model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

display(ipd.Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"]))
print(f"Reference: {sample['text']}")
print(f"Result: {transcription[0]}")

# Function to measure CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

# Function to measure performance with CPU usage and calculate WER (case-insensitive)
def measure_perf(model, sample, processor, true_transcription, n=10):
    timers = []
    cpu_usages = []
    input_features = extract_input_features(sample)

    for _ in tqdm(range(n), desc="Measuring performance"):
        start = time.perf_counter()
        
        # Measure CPU usage before inference
        start_cpu = get_cpu_usage()

        # Run inference
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Measure CPU usage after inference
        end_cpu = get_cpu_usage()

        end = time.perf_counter()

        timers.append(end - start)
        cpu_usages.append(end_cpu - start_cpu)

    # Convert both transcriptions to lowercase for case-insensitive WER
    transcription_wer = wer(true_transcription.lower(), transcription.lower())

    return np.median(timers), np.median(cpu_usages), transcription_wer

# Measure performance for both PyTorch and OpenVINO models
perf_torch, cpu_torch, wer_torch = measure_perf(pt_model, sample, processor, sample['text'])
print(f"PyTorch Model - Inference time: {perf_torch:.3f}s, CPU Usage: {cpu_torch:.2f}%, WER: {wer_torch:.4f}")

# Load the model with OpenVINO
model_path = Path(model_id_value.replace("/", "_"))
ov_config = {"CACHE_DIR": ""}

if not model_path.exists():
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id_value,
        ov_config=ov_config,
        export=True,
        compile=False,
        load_in_8bit=False,
    )
    ov_model.half()
    ov_model.save_pretrained(model_path)
else:
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_path, ov_config=ov_config, compile=False)

# Initialize OpenVINO core and device
core = ov.Core()
device_value = "AUTO"

# Compile and load the model on the selected device
ov_model.to(device_value)
ov_model.compile()

# Measure performance for the OpenVINO model
perf_ov, cpu_ov, wer_ov = measure_perf(ov_model, sample, processor, sample['text'])
print(f"OpenVINO Model - Inference time: {perf_ov:.3f}s, CPU Usage: {cpu_ov:.2f}%, WER: {wer_ov:.4f}")

# Display comparison results
print(f"Performance Comparison:\n")
print(f"PyTorch Inference Time: {perf_torch:.3f}s vs OpenVINO Inference Time: {perf_ov:.3f}s")
print(f"PyTorch CPU Usage: {cpu_torch:.2f}% vs OpenVINO CPU Usage: {cpu_ov:.2f}%")
print(f"PyTorch WER: {wer_torch:.4f} vs OpenVINO WER: {wer_ov:.4f}")
