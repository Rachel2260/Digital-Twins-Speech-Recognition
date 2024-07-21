import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import torch
import numpy as np
import json
from transformers import BertTokenizerFast


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

# file_path = 'data/tagged_donor_sentences_2000.conll'
file_path = 'data/labeled_sentences.conll'
sentences, labels = read_conll_data(file_path)
data = {'words': sentences, 'labels': labels}
df = pd.DataFrame(data)

dataset = Dataset.from_pandas(df)

train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

print("Loading saved model for inference...")

model = BertForTokenClassification.from_pretrained("./final_model_test")
tokenizer = BertTokenizerFast.from_pretrained("./final_model_test")

ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy='simple', device=0 if torch.cuda.is_available() else -1)

print("Running inference on test set...")
test_texts = [" ".join(words) for words in test_dataset['words']]
test_results = [ner_pipe(text) for text in test_texts]

def convert_float32_to_float(data):
    if isinstance(data, dict):
        return {k: convert_float32_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_float32_to_float(item) for item in data]
    elif isinstance(data, np.float32):
        return float(data)
    else:
        return data

test_results = convert_float32_to_float(test_results)

for i, text in enumerate(test_texts):
    print(f"\nText: {text}")
    print("Entities:")
    for entity in test_results[i]:
        print(f"  Entity: {entity['word']}, Type: {entity['entity_group']}, Confidence: {entity['score']:.2f}")


with open('test_set_results_2.json', 'w') as f:
    json.dump(test_results, f, indent=4)
print("Test set results saved to test_set_results_2.json")
