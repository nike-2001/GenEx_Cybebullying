# Import necessary libraries
import csv  # For reading and writing CSV files
import re  # For regular expressions
import pickle  # For serializing and deserializing Python objects
import time  # For time-related functions
import pandas as pd  # For data manipulation and analysis
from comet import Comet  # Custom library for generating commonsense knowledge
import torch  # For PyTorch operations
import nltk  # For text preprocessing and tokenization
from sklearn.model_selection import train_test_split  # For splitting datasets

# Define the relations to extract using the Comet model
relations = ["xNeed", "xWant", "xAttr", "xEffect", 'xReact', 'oEffect', 'oReact', 'oWant']

# Define word pairs for expanding contractions in sentences
word_pairs = {
    "it's": "it is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "you'd": "you would",
    "you're": "you are",
    "you'll": "you will",
    "i'm": "i am",
    "they're": "they are",
    "that's": "that is",
    "what's": "what is",
    "couldn't": "could not",
    "i've": "i have",
    "we've": "we have",
    "can't": "cannot",
    "i'd": "i would",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "there's": "there is",
    "there're": "there are",
}

# Function to preprocess a sentence
def process_sent(sentence):
    """
    Preprocess a sentence by lowercasing, expanding contractions, and tokenizing.
    Args:
        sentence (str): The input sentence.
    Returns:
        list: Tokenized words.
    """
    sentence = sentence.lower()  # Convert to lowercase
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)  # Replace contractions with full forms
    sentence = nltk.word_tokenize(sentence)  # Tokenize the sentence
    return sentence

# Function to generate commonsense knowledge using Comet
def get_commonsense(comet, item):
    """
    Generate commonsense knowledge for a given input.
    Args:
        comet (Comet): Pretrained Comet model object.
        item (list): List of tokenized words.
    Returns:
        list: Generated commonsense knowledge for each relation.
    """
    cs_list = []
    input_event = " ".join(item)  # Combine tokens into a single string
    for rel in relations:
        cs_res = comet.generate(input_event, rel)  # Generate commonsense for the relation
        cs_list.append(cs_res)
    return cs_list

# Function to convert a list to a space-separated string
def listToString(s):
    """
    Convert a list of words into a space-separated string.
    Args:
        s (list): List of words.
    Returns:
        str: Space-separated string.
    """
    str1 = ""
    for i in range(len(s)):
        if i == 0:
            str1 += s[i]
        else:
            str1 = str1 + " " + s[i]
    return str1

# Specify the CSV file name containing data
filename = "Complaint data annotation (explain)_updated - cd.csv"

# Initialize lists to store fields (columns) and rows (data)
fields = []
rows = []

# Read the CSV file
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)  # Create a CSV reader object
    fields = next(csvreader)  # Extract field names from the first row
    for row in csvreader:
        rows.append(row)  # Append each data row to the rows list
    print("Total no. of rows: %d" % (csvreader.line_num))  # Print total number of rows

# Print field names and first few rows for debugging
print('Field names are:' + ', '.join(field for field in fields))
print('\nFirst 5 rows are:\n')
cnt = 0
data = []
for row in rows:
    if len(row[1]) > 0:  # Check if the row contains valid data
        cnt += 1
        print(row[1])  # Print the data for debugging
        data.append(row)  # Add valid rows to the `data` list
print(cnt)

# Clean up whitespace in the data
for i in range(len(data)):
    data[i][1] = " ".join(data[i][1].split())  # Remove extra spaces

# Further cleaning to remove tokens starting with '@'
newdata = []
for i in range(len(data)):
    newlist = []
    for j in range(len(data[i][1].split())):
        if data[i][1].split()[j][0] == '@':
            continue
        else:
            newlist.append(data[i][1].split()[j])
    newdata.append(listToString(newlist))

# Replace cleaned data back into the original dataset
for i in range(len(data)):
    data[i][1] = newdata[i]

# Remove URLs from the data
for i in range(len(data)):
    data[i][0] = re.sub(r'http\S+', '', data[i][0])

# Create a simplified data format for processing
newdata = []
for i in range(len(data)):
    newdata.append(data[i])
for i in range(len(data)):
    newdata[i] = data[i][1:3]  # Extract relevant columns
    newdata[i][2:] = data[i][-2:]

data = newdata

# Initialize device and Comet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
comet = Comet('COMEDMODELPATH', device)

# Process the data using the Comet model
for i in range(len(data)):
    # Convert labels to meaningful text
    if data[i][1] == '1':
        data[i][1] = 'Complaint'
    else:
        data[i][1] = 'NonComplaint'

    # Combine features into a structured format
    data[i][1] = data[i][3] + ' < ' + data[i][2] + ' > ' + '< ' + data[i][1] + ' >'
    data[i] = data[i][:2]

    sent = process_sent(data[i][0])  # Preprocess the text
    CS = get_commonsense(comet, sent)  # Generate commonsense knowledge

    # Append commonsense knowledge to the data
    data[i].append(CS[0][0] + ' ' + CS[1][0])

# Convert data to a Pandas DataFrame for easier manipulation
df = pd.DataFrame(data, columns=['text', 'label', 'CS'])

# Split data into train, test, and validation sets
train_data, test_data = train_test_split(df, test_size=0.1, random_state=11)
train_data, eval_data = train_test_split(train_data, test_size=0.05, random_state=11)

# Print a preview of each dataset
print(train_data.head())
print(eval_data.head())
print(test_data.head())
