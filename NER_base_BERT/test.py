import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import torch
import numpy as np
import json
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
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

print("Loading saved model for inference...")

model = BertForTokenClassification.from_pretrained("./final_model")
tokenizer = BertTokenizerFast.from_pretrained("./final_model")

ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

print("Running inference on test set...")
test_texts = [" ".join(words) for words in test_dataset['words']]
test_labels = test_dataset['labels']

# Perform NER on each sentence in the test set
test_results = [ner_pipe(text) for text in test_texts]

# Merging subwords with preceding tokens and keeping 'O' for non-entity tokens
def merge_subwords(tokens, entities):
    merged_tokens = []
    merged_entities = []
    current_token = ""
    current_entity = "O"  # Default to 'O' for non-entity tokens
    current_confidence = 1.0

    for token, entity in zip(tokens, entities):
        if token.startswith("##"):
            current_token += token[2:]  # Merge subword with the previous token
        else:
            if current_token:  # If a current_token is being built, add it to the result
                merged_tokens.append(current_token)
                merged_entities.append((current_entity, current_confidence))  # Track the entity type and confidence
            current_token = token  # Start a new token
            current_entity = entity['entity'] if 'entity' in entity else 'O'
            current_confidence = entity['score'] if 'score' in entity else 1.0
    if current_token:  # Add the last token
        merged_tokens.append(current_token)
        merged_entities.append((current_entity, current_confidence))

    return merged_tokens, merged_entities

# Aligning predictions and true labels
def align_predictions_and_labels(test_results, test_labels):
    preds = []
    true_labels_merged = []

    for i, (result, true_label) in enumerate(zip(test_results, test_labels)):
        predicted_tokens = [entity['word'] for entity in result]
        predicted_entities = [{'entity': entity.get('entity_group', entity.get('entity', 'O')), 'score': entity['score']} for entity in result]

        # Merge subwords
        merged_tokens, merged_entities = merge_subwords(predicted_tokens, predicted_entities)

        # Align predictions with true labels
        sentence_preds = [entity for entity, _ in merged_entities]

        # For each token in the true label, if it is an 'O', it should remain as 'O' in the predicted output
        aligned_preds = []
        index = 0
        for token, label in zip(test_dataset['words'][i], true_label):
            if label == 'O':
                aligned_preds.append('O')
            else:
                # Align the predicted entity with the true entity
                aligned_preds.append(sentence_preds[index] if index < len(sentence_preds) else 'O')
                index += 1
        
        preds.append(aligned_preds)
        true_labels_merged.append(true_label)

    return preds, true_labels_merged

# Align predictions and true labels
predictions, true_labels_merged = align_predictions_and_labels(test_results, test_dataset['labels'])

# Now calculate metrics using the seqeval library
precision = precision_score(true_labels_merged, predictions, zero_division=1)
recall = recall_score(true_labels_merged, predictions, zero_division=1)
f1 = f1_score(true_labels_merged, predictions, zero_division=1)
accuracy = accuracy_score(true_labels_merged, predictions)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Output sample predictions
for i in range(5):  # Output 5 test examples
    print(f"\nText: {' '.join(test_dataset['words'][i])}")
    print(f"True Labels: {test_dataset['labels'][i]}")
    print(f"Predicted Labels: {predictions[i]}")

# Generate the classification report
print("\nClassification Report:")
report = classification_report(true_labels_merged, predictions, zero_division=1)
print(report)