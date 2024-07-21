from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset
import openvino as ov
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
import time
import numpy as np
from tqdm import tqdm
from itertools import islice
import gc
import shutil
import nncf
from pathlib import Path
from contextlib import contextmanager
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
model_id_value = model_ids[model_type_value][0]

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

# Load the model with OpenVINO and perform quantization if enabled
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

# Perform inference with the OpenVINO model
predicted_ids = ov_model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

display(ipd.Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"]))
print(f"Reference: {sample['text']}")
print(f"Result: {transcription[0]}")

# Function to measure performance
def measure_perf(model, sample, n=10):
    timers = []
    input_features = extract_input_features(sample)
    for _ in tqdm(range(n), desc="Measuring performance"):
        start = time.perf_counter()
        model.generate(input_features)
        end = time.perf_counter()
        timers.append(end - start)
    return np.median(timers)

# Measure performance for both PyTorch and OpenVINO models
perf_torch = measure_perf(pt_model, sample)
perf_ov = measure_perf(ov_model, sample)

print(f"Mean torch {model_id_value} generation time: {perf_torch:.3f}s")
print(f"Mean openvino {model_id_value} generation time: {perf_ov:.3f}s")
print(f"Performance {model_id_value} openvino speedup: {perf_torch / perf_ov:.3f}")

# Prepare SRT format
def format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    return (f"{hours}:" if hours > 0 else "00:") + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def prepare_srt(transcription):
    segment_lines = []
    for idx, segment in enumerate(transcription["chunks"]):
        segment_lines.append(str(idx + 1) + "\n")
        timestamps = segment["timestamp"]
        time_start = format_timestamp(timestamps[0])
        time_end = format_timestamp(timestamps[1])
        time_str = f"{time_start} --> {time_end}\n"
        segment_lines.append(time_str)
        segment_lines.append(segment["text"] + "\n\n")
    return segment_lines

# Quantization
to_quantize = True

# Function to collect calibration dataset
def collect_calibration_dataset(ov_model: OVModelForSpeechSeq2Seq, calibration_dataset_size: int):
    encoder_calibration_data = []
    decoder_calibration_data = []
    ov_model.encoder.request = InferRequestWrapper(ov_model.encoder.request, encoder_calibration_data, apply_caching=True)
    ov_model.decoder_with_past.request = InferRequestWrapper(ov_model.decoder_with_past.request,
                                                             decoder_calibration_data,
                                                             apply_caching=True)
    try:
        calibration_dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
        for sample in tqdm(islice(calibration_dataset, calibration_dataset_size), desc="Collecting calibration data", total=calibration_dataset_size):
            input_features = extract_input_features(sample)
            ov_model.generate(input_features)
    finally:
        ov_model.encoder.request = ov_model.encoder.request.request
        ov_model.decoder_with_past.request = ov_model.decoder_with_past.request.request
    return encoder_calibration_data, decoder_calibration_data

if to_quantize:
    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)

    from optimum.intel.openvino.quantization import InferRequestWrapper

    CALIBRATION_DATASET_SIZE = 50

    def quantize(ov_model: OVModelForSpeechSeq2Seq, calibration_dataset_size: int):
        quantized_model_path = Path(f"{model_path}_quantized")
        if not quantized_model_path.exists():
            encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(
                ov_model, calibration_dataset_size
            )
            print("Quantizing encoder")
            quantized_encoder = nncf.quantize(
                ov_model.encoder.model,
                nncf.Dataset(encoder_calibration_data),
                subset_size=len(encoder_calibration_data),
                model_type=nncf.ModelType.TRANSFORMER,
                advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.50)
            )
            ov.save_model(quantized_encoder, quantized_model_path / "openvino_encoder_model.xml")
            del quantized_encoder
            del encoder_calibration_data
            gc.collect()

            print("Quantizing decoder with past")
            quantized_decoder_with_past = nncf.quantize(
                ov_model.decoder_with_past.model,
                nncf.Dataset(decoder_calibration_data),
                subset_size=len(decoder_calibration_data),
                model_type=nncf.ModelType.TRANSFORMER,
                advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.95)
            )
            ov.save_model(quantized_decoder_with_past, quantized_model_path / "openvino_decoder_with_past_model.xml")
            del quantized_decoder_with_past
            del decoder_calibration_data
            gc.collect()

            # Copy the config file and the first-step-decoder manually
            shutil.copy(model_path / "config.json", quantized_model_path / "config.json")
            shutil.copy(model_path / "openvino_decoder_model.xml", quantized_model_path / "openvino_decoder_model.xml")
            shutil.copy(model_path / "openvino_decoder_model.bin", quantized_model_path / "openvino_decoder_model.bin")

        quantized_ov_model = OVModelForSpeechSeq2Seq.from_pretrained(quantized_model_path, ov_config=ov_config, compile=False)
        quantized_ov_model.to(device_value)
        quantized_ov_model.compile()
        return quantized_ov_model

    ov_quantized_model = quantize(ov_model, CALIBRATION_DATASET_SIZE)

    # Load a sample dataset
    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True
    )
    sample = dataset[0]
    input_features = extract_input_features(sample)

    predicted_ids = ov_model.generate(input_features)
    transcription_original = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    predicted_ids = ov_quantized_model.generate(input_features)
    transcription_quantized = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    display(ipd.Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"]))
    print(f"Original : {transcription_original[0]}")
    print(f"Quantized: {transcription_quantized[0]}")

    # Function to measure transcription time and accuracy
    def time_fn(obj, fn_name, time_list):
        original_fn = getattr(obj, fn_name)

        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = original_fn(*args, **kwargs)
            end_time = time.perf_counter()
            time_list.append(end_time - start_time)
            return result

        setattr(obj, fn_name, wrapper)

    @contextmanager
    def time_measurement():
        global MEASURE_TIME
        try:
            MEASURE_TIME = True
            yield
        finally:
            MEASURE_TIME = False

    def calculate_transcription_time_and_accuracy(ov_model, test_samples):
        encoder_infer_times = []
        decoder_with_past_infer_times = []
        whole_infer_times = []
        time_fn(ov_model, "generate", whole_infer_times)
        time_fn(ov_model.encoder, "forward", encoder_infer_times)
        time_fn(ov_model.decoder_with_past, "forward", decoder_with_past_infer_times)

        ground_truths = []
        predictions = []
        for data_item in tqdm(test_samples, desc="Measuring performance and accuracy"):
            input_features = extract_input_features(data_item)

            with time_measurement():
                predicted_ids = ov_model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            ground_truths.append(data_item["text"])
            predictions.append(transcription[0])

        word_accuracy = (1 - wer(ground_truths, predictions, reference_transform=wer_standardize,
                                hypothesis_transform=wer_standardize)) * 100
        mean_whole_infer_time = sum(whole_infer_times)
        mean_encoder_infer_time = sum(encoder_infer_times)
        mean_decoder_with_time_infer_time = sum(decoder_with_past_infer_times)
        return word_accuracy, (mean_whole_infer_time, mean_encoder_infer_time, mean_decoder_with_time_infer_time)

    TEST_DATASET_SIZE = 50
    test_dataset = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True, trust_remote_code=True)
    test_dataset = test_dataset.shuffle(seed=42).take(TEST_DATASET_SIZE)
    test_samples = [sample for sample in test_dataset]

    accuracy_original, times_original = calculate_transcription_time_and_accuracy(ov_model, test_samples)
    accuracy_quantized, times_quantized = calculate_transcription_time_and_accuracy(ov_quantized_model, test_samples)
    print(f"Encoder performance speedup: {times_original[1] / times_quantized[1]:.3f}")
    print(f"Decoder with past performance speedup: {times_original[2] / times_quantized[2]:.3f}")
    print(f"Whole pipeline performance speedup: {times_original[0] / times_quantized[0]:.3f}")
    print(f"Whisper transcription word accuracy. Original model: {accuracy_original:.2f}%. Quantized model: {accuracy_quantized:.2f}%.")
    print(f"Accuracy drop: {accuracy_original - accuracy_quantized:.2f}%.")
