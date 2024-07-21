import os
import re
import numpy as np
import speech_recognition as sr
import whisper
import torch
import json
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from transformers import pipeline

def main():
    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    audio_model = whisper.load_model("medium.en")

    record_timeout = 3
    phrase_timeout = 4

    transcription = ['']

    ner_pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    def add_a_case():
        print("**** Open Add a Case page ****")
    
    def start_writing():
        print("**** Start writing the form ****")
        if capturing:
            process_ner(" ".join(transcription))
    
    def process_ner(text):
        result = ner_pipe(text)

        def extract_info(entities):
            info = {
                'ID': None,
                'DoD': None,
                'Age': None,
                'Cause of death': None
            }

            for entity in entities:
                if entity['entity_group'] == 'DETAILED_DESCRIPTION':
                    info['ID'] = entity['word']
                elif entity['entity_group'] == 'DATE':
                    info['DoD'] = entity['word']
                elif entity['entity_group'] == 'LAB_VALUE':
                    info['Age'] = entity['word']
                elif entity['entity_group'] == 'DISEASE_DISORDER':
                    info['Cause of death'] = entity['word']

            return info

        extracted_info = extract_info(result)
        with open('extracted_real_time_info.json', 'w') as json_file:
            json.dump(extracted_info, json_file, indent=4)

        print("Extracted information saved to extracted_real_time_info.json")
    
    capturing = False

    while True:
        try:
            now = datetime.now()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if phrase_complete or not transcription:
                    transcription.append(text)
                else:
                    transcription[-1] += " " + text

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                    if re.search(r"add\s+a\s+case", line, re.IGNORECASE):
                        add_a_case()
                    if re.search(r"start\s+writing", line, re.IGNORECASE):
                        capturing = True
                        start_writing()
                    if re.search(r"stop\s+writing", line, re.IGNORECASE):
                        capturing = False
                print('', end='', flush=True)

            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()
