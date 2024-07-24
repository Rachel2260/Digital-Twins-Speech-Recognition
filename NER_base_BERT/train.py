import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import classification_report, f1_score
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

model_name = "bert-large-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
label_list = ["O", "B-ID", "I-ID", "B-DoD", "I-DoD", "B-Age", "I-Age", "B-Cause_of_Death", "I-Cause_of_Death"]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id)

model.to(device)

# preprocess data
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['words'],
        truncation=True,
        padding=True,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_list.index(label[word_idx]))
            else:
                label_ids.append(label_list.index("I-"+label[word_idx][2:]) if label[word_idx] != 'O' else label_list.index('O'))
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

# training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    return {
        "f1": f1_score(true_labels, true_predictions),
        "report": classification_report(true_labels, true_predictions)
    }

# make sure all tensors are on the same device
def move_to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        inputs = move_to_device(inputs, device)
        outputs = model(**inputs)
        loss = outputs.loss
        loss = loss.to(device)
        loss.backward()
        return loss.detach().cpu()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        inputs = self._prepare_inputs(inputs)
        inputs = move_to_device(inputs, device)
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        if loss is not None:
            loss = loss.to(device)
        logits = logits.to(device)
        labels = inputs.get("labels").to(device)
        return (loss, logits, labels)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# evaluation
results = trainer.evaluate()
print(results)

# save the last model
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
