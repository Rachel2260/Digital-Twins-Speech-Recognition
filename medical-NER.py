import json
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load the NER pipeline
pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')

# Load the content of transcription.txt
with open('transcription.txt', 'r') as file:
    text = file.read()

# Use the model to process the text
result = pipe(text)
print(result)

# Function to extract specific information
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

# Extract information from the result
extracted_info = extract_info(result)

# Save the extracted information as a JSON file
with open('extracted_info.json', 'w') as json_file:
    json.dump(extracted_info, json_file, indent=4)

print("Extracted information saved to extracted_info.json")
