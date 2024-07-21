import spacy
import json
import re
from datetime import datetime

# load spaCy model
nlp = spacy.load('en_core_web_sm')

# define keywords for information extraction
id_keywords = ["ID", "ID number", "donor's ID", "identification number", "identity number", "identity ID", "identity", "donor ID", "donor number", "donor's number", "donor's identity", "donor's identity number", "donor's identification number"]
dod_keywords = ["date of death", "DoD", "death date", "date of passing", "date of decease"]
age_keywords = ["age", "years old", "old", "age of the donor", "age of the deceased", "age of the person"]
cause_keywords = ["cause of death", "reason of death", "manner of death", "cause of decease", "reason of decease", "manner of decease", "cause", "reason", "manner", "because of"]

# read the transcription file
with open('transcription.txt', 'r') as file:
    text = file.read()

# process the text with spaCy
doc = nlp(text)

# extract information using keywords and patterns
def extract_info(text, keywords, pattern):
    for keyword in keywords:
        full_pattern = rf'\b{keyword}\b.*?{pattern}'
        match = re.search(full_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

# preprocess the date
def preprocess_date(date_str):
    # remove suffixes from day
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
    date_formats = ['%B %d, %Y', '%B %d %Y', '%d of %B %Y', '%d of %B, %Y', '%d %B %Y', '%d %B, %Y', '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%d %m %Y', '%d %m, %Y', '%m %d %Y', '%m %d, %Y', '%m %d %Y']
    for fmt in date_formats:
        try:
            # transform date to DD/MM/YYYY format
            formatted_date = datetime.strptime(date_str, fmt).strftime('%d/%m/%Y')
            return formatted_date
        except ValueError:
            continue
    return None

# extract information and format it
id_number = extract_info(text, id_keywords, r'(\d+)')
date_of_death = extract_info(text, dod_keywords, r'(\d{1,2}(?:st|nd|rd|th)? of \w+,? \d{4})')
if date_of_death:
    date_of_death = preprocess_date(date_of_death)
age = extract_info(text, age_keywords, r'(\d+)')
cause_of_death = extract_info(text, cause_keywords, r'(?:is\s+)?([\w\s]+)')
if cause_of_death:
    cause_of_death = re.sub(r'^is\s+', '', cause_of_death).strip()

# create a dictionary with the extracted information
data = {
    "ID": int(id_number) if id_number else None,
    "DoD": date_of_death,
    "Age": int(age) if age else None,
    "Cause of Death": cause_of_death.strip() if cause_of_death else None
}

# save the extracted information to a JSON file
output_file = 'output.json'
with open(output_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Information extracted and saved to {output_file}")
