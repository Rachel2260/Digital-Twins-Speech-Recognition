import random
from datetime import datetime, timedelta
import pandas as pd

# Function to generate random date in different formats
def random_date():
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2024, 1, 1)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    
    date_formats = [
        "%B %d, %Y",   # January 01, 2023
        "%b %d, %Y",   # Jan 01, 2023
        "%m/%d/%Y",    # 01/01/2023
        "%-m/%-d/%Y",  # 1/1/2023
        "%d %B %Y",    # 01 January 2023
        "%d %b %Y",    # 01 Jan 2023
        "%-d %B %Y",   # 1 January 2023
        "%-d %b %Y",   # 1 Jan 2023
        "%Y-%m-%d",    # 2023-01-01
        "%Y/%m/%d",    # 2023/01/01
        "%B %-d, %Y",  # January 1, 2023
        "%b %-d, %Y",  # Jan 1, 2023
        "%Y-%m-%d",    # 2023-01-01
        "%B %-d %Y",   # January 1 2023
        "%b %-d %Y",   # Jan 1 2023
        "%Y %B %d",    # 2023 January 01
        "%Y %b %d",    # 2023 Jan 01
    ]
    
    return random_date.strftime(random.choice(date_formats))
# Function to generate random ID
def random_id():
    return str(random.randint(1000, 999999999))

# Function to generate random age
def random_age():
    return str(random.randint(1, 100))

# Function to generate random cause of death from ICD data
def random_cause_of_death(icd_data):
    return icd_data.sample(1).iloc[0]['Title'].lower()

# Function to generate sentences with varied content
def generate_varied_sentences_batch_enhanced(num_sentences, icd_data):
    varied_sentences = []

    for _ in range(num_sentences):
        donor_id = random_id()
        date_of_death = random_date()
        age = random_age()
        cause_of_death = random_cause_of_death(icd_data)
        
        options = [
            f"Donor ID {donor_id}.",
            f"Donor ID was {donor_id}.",
            f"ID {donor_id}.",
            f"ID is {donor_id}.",
            f"Donor's ID was {donor_id}.",
            f"The date of death was {date_of_death}.",
            f"The donor died on {date_of_death}.",
            f"The donor passed away on {date_of_death}.",
            f"The donor was {age} years old.",
            f"The donor was aged {age}.",
            f"The donor was {age} years of age.",
            f"The cause of death was {cause_of_death}.",
            f"The donor passed away due to {cause_of_death}.",
            f"The donor died due to {cause_of_death}.",
            f"Donor ID {donor_id} and the date of death was {date_of_death}.",
            f"Donor with ID {donor_id} died on {date_of_death}.",
            f"Donor's ID was {donor_id} who died on {date_of_death}.",
            f"Donor passed away on {date_of_death} and donor's ID was {donor_id}.",
            f"On {date_of_death}, donor ID {donor_id} passed away.",
            f"On {date_of_death}, donor ID {donor_id} died.",
            f"On {date_of_death}, donor with ID {donor_id} passed away.",
            f"Donor ID {donor_id}, aged {age}.",
            f"ID {donor_id}, age {age}.",
            f"Donor ID was {donor_id} and the donor was {age} years old.",
            f"Donor, ID {donor_id}, was {age} years old.",
            f"Donor ID {donor_id} and the donor was {age} years old.",
            f"Donor ID {donor_id} and the cause of death was {cause_of_death}.",
            f"Due to {cause_of_death}, donor ID {donor_id} passed away.",
            f"Due to {cause_of_death}, donor ID {donor_id} died.",
            f"Due to {cause_of_death}, donor with ID {donor_id} passed away.",
            f"Because of {cause_of_death}, donor ID {donor_id} passed away.",
            f"Because of {cause_of_death}, donor ID {donor_id} died.",
            f"Because of {cause_of_death}, donor with ID {donor_id} passed away.",
            f"Donor with ID {donor_id} passed away due to {cause_of_death}.",
            f"The date of death was {date_of_death} and the donor was {age} years old.",
            f"Donor passed away on {date_of_death} and was {age} years old.",
            f"Donor died on {date_of_death} and was {age} years old.",
            f"The date of death was {date_of_death} and the cause of death was {cause_of_death}.",
            f"The donor was {age} years old and the cause of death was {cause_of_death}.",
            f"Donor ID {donor_id}, the date of death was {date_of_death}, and the donor was {age} years old.",
            f"Donor ID {donor_id}, the date of death was {date_of_death}, and the cause of death was {cause_of_death}.",
            f"Donor ID {donor_id}, the donor was {age} years old, and the cause of death was {cause_of_death}.",
            f"The date of death was {date_of_death}, the donor was {age} years old, and the cause of death was {cause_of_death}.",
            f"ID {donor_id}, death date {date_of_death}, age {age}.",
            f"Donor aged {age}, ID {donor_id}, died on {date_of_death}.",
            f"Died on {date_of_death}, ID {donor_id}, cause {cause_of_death}.",
            f"ID {donor_id}, age {age}, died due to {cause_of_death}.",
            f"Death date {date_of_death}, ID {donor_id}, age {age}.",
        ]
        
        varied_sentences.append(random.choice(options))
    
    return varied_sentences

# Load the ICD data from the provided Excel file
icd_df = pd.read_excel('SimpleTabulation-ICD-11-MMS-en.xlsx')

# Generate 1000 sentences in batches
enhanced_varied_sentences = []
batch_size = 100
num_batches = 10

for _ in range(num_batches):
    batch_sentences = generate_varied_sentences_batch_enhanced(batch_size, icd_df)
    enhanced_varied_sentences.extend(batch_sentences)

# Create a DataFrame for all sentences
enhanced_varied_sentences_df = pd.DataFrame(enhanced_varied_sentences, columns=["Sentence"])

# Save to CSV
csv_file_path = 'varied_donor_sentences_1-3.csv'
enhanced_varied_sentences_df.to_csv(csv_file_path, index=False)

print(f"CSV file saved at: {csv_file_path}")