# -*- coding: utf-8 -*-
# This script performs detokenization, converting tokenized text back into a readable format.

import sys
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk

# Extend the default punctuation set with additional characters specific to the dataset
punc = string.punctuation  # Default punctuation characters
punc += "n't，《。》、？；：‘”【「】」、|·~！@#￥%……&*（）-——+='s'm'll''``"

# Initialize an empty list to store sequences from the input file
seqs = []

# Read the tokenized text file provided as the first command-line argument
with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        seqs.append(line.strip())  # Strip whitespace and append each line to the sequence list

# Open the output file specified as the second command-line argument
with open(sys.argv[2], 'w') as f:
    for line in seqs:
        # Tokenize the line using NLTK's tokenizer
        line = nltk.word_tokenize(line.strip())

        # Detokenize the tokenized line into a readable sentence
        line = TreebankWordDetokenizer().detokenize(line)

        # Split the detokenized line into individual tokens
        tokens = line.split()
        line = ''

        # Reconstruct the line by appending tokens
        for token in tokens:
            if token not in punc:  # If the token is not punctuation, prepend a space
                line += (' ' + token)
            else:  # Directly append punctuation without a preceding space
                line += token

        # Write the detokenized and formatted line to the output file
        f.write(line.strip() + '\n')
