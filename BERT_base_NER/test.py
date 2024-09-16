import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import torch
import time
import psutil
import GPUtil
import numpy as np
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

def read_conll_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r') as f:
        sentence = []
        label = []
        line_number = 0
        for line in f:
            line_number += 1
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                parts = line.strip().split()
                if len(parts) == 2:
                    token, tag = parts
                    sentence.append(token)
                    label.append(tag)
                else:
                    print(f"Skipping line {line_number}: {line.strip()}")
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

file_path = 'data/labeled_sentences.conll'
sentences, labels = read_conll_data(file_path)
data = {'words': sentences, 'labels': labels}
df = pd.DataFrame(data)

dataset = Dataset.from_pandas(df)

train_test_split = dataset.train_test_split(test_size=0.2)
test_dataset = train_test_split['test']

print("Loading saved model for inference...")

model = BertForTokenClassification.from_pretrained("./final_model")
tokenizer = BertTokenizerFast.from_pretrained("./final_model")

ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Measure inference time and resource usage
start_time = time.time()

cpu_before = psutil.cpu_percent(interval=None)
mem_before = psutil.virtual_memory().used / (1024 ** 3)  # GB
gpus = GPUtil.getGPUs()
gpu_before = gpus[0].load * 100 if gpus else None

# Running inference on test set
test_texts = [" ".join(words) for words in test_dataset['words']]
test_results = [ner_pipe(text) for text in test_texts]

cpu_after = psutil.cpu_percent(interval=None)
mem_after = psutil.virtual_memory().used / (1024 ** 3)  # GB
end_time = time.time()
total_time = end_time - start_time

gpus = GPUtil.getGPUs()
gpu_after = gpus[0].load * 100 if gpus else None

print(f"Inference Time: {total_time:.4f} seconds")
print(f"CPU Usage before: {cpu_before}% after: {cpu_after}%")
print(f"Memory Usage before: {mem_before:.2f}GB after: {mem_after:.2f}GB")

if gpu_before and gpu_after:
    print(f"GPU Usage before: {gpu_before}% after: {gpu_after}%")

# Now calculate metrics using the seqeval library
predictions = [[x['entity'] for x in result] for result in test_results]
precision = precision_score(test_dataset['labels'], predictions, zero_division=1)
recall = recall_score(test_dataset['labels'], predictions, zero_division=1)
f1 = f1_score(test_dataset['labels'], predictions, zero_division=1)
accuracy = accuracy_score(test_dataset['labels'], predictions)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Output sample predictions
for i in range(5):  # Output 5 test examples
    print(f"\nText: {' '.join(test_dataset['words'][i])}")
    print(f"True Labels: {test_dataset['labels'][i]}")
    print(f"Predicted Labels: {predictions[i]}")
