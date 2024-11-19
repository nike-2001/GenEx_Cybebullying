# -*- coding: utf-8 -*-
# This script tokenizes text using NLTK's word tokenizer, with an option to lowercase the input text.

import sys
import nltk  # Import the Natural Language Toolkit (NLTK) for tokenization

# Read input text file line by line
fin = open(sys.argv[1], 'r').readlines()

# Open the output file for writing tokenized text
with open(sys.argv[2], 'w') as f:
    for line in fin:
        # Check if the third command-line argument is 'True'
        if sys.argv[3] == 'True':
            # Tokenize the text and convert to lowercase
            line = nltk.word_tokenize(line.strip().lower())
        else:
            # Tokenize the text without changing the case
            line = nltk.word_tokenize(line.strip())
        # Write the tokenized text to the output file, separated by spaces
        f.write(' '.join(line) + '\n')
