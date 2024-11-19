# -*- coding: utf-8 -*-
# This script calculates the BLEURT scores for evaluating the quality of generated text against reference texts.

import sys  # For command-line argument parsing
from bleurt import score  # BLEURT library for scoring the quality of text generation

# Specify the path to the BLEURT model checkpoint
checkpoint = '../checkpoints/bleurt-base-128'
scorer = score.BleurtScorer(checkpoint)  # Initialize the BLEURT scorer with the specified checkpoint


def cal_bleurt(file0, file1):
    """
    Calculate BLEURT scores between hypotheses and references.

    Args:
        file0 (str): Path to the file containing hypothesis texts (one per line).
        file1 (str): Base path to the files containing reference texts (appends numbers 0-3).

    Returns:
        List[float]: List of BLEURT scores for the hypothesis-reference pairs.
    """
    scores = []  # List to store BLEURT scores
    with open(file0, 'r') as fin:
        hyps = []  # List to store hypotheses
        for line in fin.readlines():
            hyps.append(line.strip())  # Read and strip whitespace from each line

    # Iterate over 4 reference files (file1 + {0, 1, 2, 3})
    for i in range(4):
        with open(file1 + str(i), 'r') as fin:
            refs = []  # List to store references
            for line in fin.readlines():
                refs.append(line.strip())  # Read and strip whitespace from each line
            # Calculate BLEURT scores for the current set of references and hypotheses
            scores.extend(scorer.score(refs, hyps))
    return scores


# Collect BLEURT scores from two sets of input files provided via command-line arguments
scores = []
scores.extend(cal_bleurt(sys.argv[1], sys.argv[3]))  # Compute scores for the first set of files
scores.extend(cal_bleurt(sys.argv[2], sys.argv[4]))  # Compute scores for the second set of files

# Print the average BLEURT score
print('The average BLEURT score is {}'.format(sum(scores) / len(scores)))
