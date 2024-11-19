import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import cuda
from torch.nn import CrossEntropyLoss
import nltk

# Importing custom model and tokenizer classes
from model import BartModel
from model import BartForMaskedLM
from transformers import BartTokenizer
from transformers.modeling_bart import make_padding_mask

# Importing utility functions for optimization and evaluation
from utils.optim import ScheduledOptim
from utils.helper import optimize, evaluate
from utils.helper import cal_sc_loss, cal_bl_loss
from utils.dataset import read_data, BARTIterator
import pickle

# Set CUDA launch blocking to aid debugging GPU operations
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Set the CUDA device to be used for PyTorch computations
torch.cuda.set_device(2)  # Use CUDA device ID 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if CUDA is available
print(device)  # Output the device being used

# Load the BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Load a pre-trained BART model
model = BartModel.from_pretrained("facebook/bart-base")  # Load the base BART model
model.config.output_past = True  # Enable caching of past key-value states for efficiency
model = BartForMaskedLM.from_pretrained(
    "facebook/bart-base",
    config=model.config
)  # Initialize BART with a language modeling head

# Move the model to GPU and set it to evaluation mode
model.to('cuda').eval()

# Iterate over the files in the directory `SS`
directory = os.listdir('SS')
for filename in directory:
    print(filename)  # Print the filename for debugging
    if filename != '4000.chkpt':  # Skip files other than the specified checkpoint
        continue

    # Load the model weights from the checkpoint
    model.load_state_dict(torch.load('SS/' + filename))
    print('loaded')  # Confirm that the weights were loaded

    preds = []  # List to store predictions

    # Load and process the dataset
    df = pd.read_csv(
        '/DATA/sriparna/BartRLCM/pre-trained-formality-transfer/Complaint data annotation (explain)_updated - cd.csv',
        header=None
    )  # Read the CSV file without headers
    print(df.head())  # Output the first few rows for debugging
    df = df[[1, 2]]  # Extract relevant columns (assumed to be text and labels)
    df = df.iloc[1:, :]  # Skip the first row (usually headers)

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(df, test_size=0.10, random_state=42)
    test_data.rename(columns={1: 'tweet'}, inplace=True)  # Rename columns for clarity
    test_data.rename(columns={2: 'SS'}, inplace=True)

    # Output the training data for debugging
    print(train_data.head())
    print('*********')

    # Extract tweets and labels from the test data
    testtext = test_data['tweet'].tolist()
    testlabels = test_data['SS'].tolist()

    # Generate predictions for the test data
    for text in testtext:
        # Tokenize and encode the text
        src = tokenizer.encode(text, return_tensors='pt')

        # Generate predictions using the model
        generated_ids = model.generate(
            src.to(device),
            num_beams=5,  # Use beam search with 5 beams
            max_length=30  # Limit the maximum length of the generated sequence
        )

        # Decode the generated IDs back to text
        text = [
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in generated_ids
        ][0]

        # Print the generated text for debugging
        print(text)
        preds.append(text)  # Append the prediction to the list

    # Calculate accuracy by comparing predictions with labels
    acc = 0
    for i in range(len(preds)):
        print(preds[i])  # Print the predicted text
        print(testlabels[i])  # Print the true label
        if preds[i].strip() == testlabels[i].strip():  # Compare predictions and labels
            acc += 1  # Increment the accuracy counter for correct predictions

    # Print the final accuracy
    print('ACC: {}'.format(acc / len(preds)))
