import sys
import os
import unittest

def read_conll_data_from_text(text):
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            word, label = line.split() 
            current_sentence.append(word)
            current_labels.append(label)
        else:
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence = []
                current_labels = []

    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)

    return sentences, labels

class TestPreprocessing(unittest.TestCase):
    def test_read_conll_data(self):
        sample_text = """
        Donor B-ID
        123456 I-ID
        died O
        """
        # Expected output
        expected_sentences = [["Donor", "123456", "died"]]
        expected_labels = [["B-ID", "I-ID", "O"]]

        # Perform the test using hand-written data
        sentences, labels = read_conll_data_from_text(sample_text)

        # Check if the output matches the expected output
        self.assertEqual(sentences, expected_sentences)
        self.assertEqual(labels, expected_labels)
if __name__ == "__main__":
    unittest.main()
