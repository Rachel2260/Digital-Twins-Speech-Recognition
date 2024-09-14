import unittest
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForTokenClassification
import torch
import time
import os

class TestModelIntegration(unittest.TestCase):
    def setUp(self):
        # Load the actual pre-trained models you are using in the WinForms application
        NER_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../NER_base_BERT/final_model"))
        whisper_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../distil-whisper/model"))
        
        self.speech_processor = AutoProcessor.from_pretrained(whisper_model_path)
        self.speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_path)

        self.ner_tokenizer = AutoTokenizer.from_pretrained(NER_model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(NER_model_path)

    def test_speech_recognition(self):
        # Generate a 1-second random audio data (mono-channel, 16000 samples)
        audio_data = torch.randn(16000)  # A 1D array representing 1 second of audio at 16 kHz
        
        # Ensure correct sampling rate is passed to the processor
        inputs = self.speech_processor(audio_data, sampling_rate=16000, return_tensors="pt")
        
        generated_ids = self.speech_model.generate(inputs["input_features"])
        transcription = self.speech_processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        self.assertIsNotNone(transcription[0])  # Check that transcription is returned

    def test_ner_model(self):
        # Test inference with NER model
        text = "John Doe, 45 years old, passed away on January 1, 2021."
        inputs = self.ner_tokenizer(text, return_tensors="pt")
        outputs = self.ner_model(**inputs)

        self.assertIsNotNone(outputs.logits)  # Check that logits are returned

    def test_incompatible_model_and_processor(self):
        # Test loading incompatible model and processor
        with self.assertRaises(Exception):
            wrong_processor = AutoProcessor.from_pretrained("some-other-model")
            wrong_model = AutoModelForSpeechSeq2Seq.from_pretrained("some-other-model")

    def test_long_text_input(self):
        # Test very long text input for NER model
        long_text = "John Doe passed away. " * 1000  # Very long text
        inputs = self.ner_tokenizer(long_text, return_tensors="pt", truncation=True)
        outputs = self.ner_model(**inputs)

        # Check that model handles long input
        self.assertIsNotNone(outputs.logits)

    def test_multilingual_input(self):
        # Test NER model with non-English input (French)
        text = "Jean Dupont est décédé le 1er janvier 2021."  # French input
        inputs = self.ner_tokenizer(text, return_tensors="pt")
        outputs = self.ner_model(**inputs)

        # Ensure the model can handle multilingual text
        self.assertIsNotNone(outputs.logits)

    def test_special_characters_input(self):
        # Test NER model with special characters input
        text = "!!@@##$$%%^^&&**(())--==++"
        inputs = self.ner_tokenizer(text, return_tensors="pt")
        outputs = self.ner_model(**inputs)

        # Check that model handles special characters
        self.assertIsNotNone(outputs.logits)

    def test_inference_time(self):
        # Test inference time for NER model
        text = "John Doe passed away on January 1st, 2021 at the age of 45."
        start_time = time.time()
        
        inputs = self.ner_tokenizer(text, return_tensors="pt")
        outputs = self.ner_model(**inputs)
        
        elapsed_time = time.time() - start_time
        # Ensure the inference time is less than 1 second
        self.assertLess(elapsed_time, 1.0)

    def test_batch_inference(self):
        # Test batch processing with NER model
        texts = ["John Doe passed away.", "Jane Doe is alive."]
        inputs = self.ner_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.ner_model(**inputs)

        # Ensure the batch size matches the number of inputs
        self.assertEqual(len(outputs.logits), len(texts))


if __name__ == '__main__':
    unittest.main()
