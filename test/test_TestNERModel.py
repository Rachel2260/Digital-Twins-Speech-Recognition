import unittest
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os

class TestNERModel(unittest.TestCase):
    def setUp(self):
        # Initialize tokenizer and model properly
        NER_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../NER_base_BERT/final_model"))

        self.tokenizer = AutoTokenizer.from_pretrained(NER_model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(NER_model_path)

    def test_ner_model_output(self):
        # Sample input
        text = "John Doe passed away on January 1st, 2021 at the age of 45."
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)

        # Check that output is not empty
        self.assertTrue(outputs.logits is not None)

        # Check the number of tokens matches expected
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        self.assertEqual(len(tokens), len(outputs.logits[0]))

    def test_empty_input(self):
        text = ""
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)

        # Print to debug the output shape
        print(f"Logits shape: {outputs.logits.shape}")
    
        # Check for CLS token (and possibly other special tokens like SEP)
        self.assertEqual(len(outputs.logits[0]), 2)  # CLS and SEP token


if __name__ == '__main__':
    unittest.main()
