# -*- coding: utf-8 -*-
# This module handles data loading, preprocessing, and batching for training and evaluation.

import random
random.seed(1024)  # Set the random seed for reproducibility
import numpy as np

import torch
import torch.utils.data


def read_data(dataset, style, max_len, prefix, tokenizer, domain=0, ratio=1.0):
    """
    Reads and preprocesses data for a specific style transfer task.

    Args:
        dataset (str): Dataset name.
        style (int): Style indicator (0 or 1) for selecting source/target files.
        max_len (int): Maximum length of sequences.
        prefix (str): Prefix for file paths.
        tokenizer: Tokenizer for text encoding.
        domain (int): Domain token to prepend to source sequences.
        ratio (float): Ratio of data to sample.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: Encoded source and target sequences.
    """
    if domain != 0:
        domain = tokenizer.encode(domain, add_special_tokens=False)[0]

    # Determine file paths based on style
    if style == 0:
        src_file = f'../data/{dataset}/{prefix}.0'
        tgt_file = f'../data/{dataset}/{prefix}.1'
    else:
        src_file = f'../data/{dataset}/{prefix}.1'
        tgt_file = f'../data/{dataset}/{prefix}.0'

    src_seq, tgt_seq = [], []
    with open(src_file, 'r') as f1, open(tgt_file, 'r') as f2:
        f1 = f1.readlines()
        f2 = f2.readlines()
        index = list(range(len(f1)))
        random.shuffle(index)  # Shuffle data indices
        index = index[:int(len(index) * ratio)]  # Sample data based on the ratio
        for i, (s, t) in enumerate(zip(f1, f2)):
            if i in index:
                # Encode and truncate sequences
                s = tokenizer.encode(s)
                t = tokenizer.encode(t)
                s = s[:min(len(s) - 1, max_len)] + s[-1:]
                t = t[:min(len(t) - 1, max_len)] + t[-1:]
                s[0] = domain  # Replace the first token with the domain token
                src_seq.append(s)
                tgt_seq.append([tokenizer.bos_token_id] + t)

    return src_seq, tgt_seq


def collate_fn(insts, pad_token_id=1):
    """
    Pads sequences in the batch to the same length.

    Args:
        insts (List[List[int]]): List of tokenized sequences.
        pad_token_id (int): Padding token ID.

    Returns:
        Tensor: Padded sequences as a PyTorch LongTensor.
    """
    max_len = max(len(inst) for inst in insts)
    max_len = max_len if max_len > 4 else 5  # Ensure minimum length of 5

    # Pad each instance to the maximum length
    batch_seq = np.array([
        inst + [pad_token_id] * (max_len - len(inst))
        for inst in insts])
    return torch.LongTensor(batch_seq)


def paired_collate_fn(insts):
    """
    Pads paired source and target sequences.

    Args:
        insts (List[Tuple[List[int], List[int]]]): List of paired sequences.

    Returns:
        Tuple[Tensor, Tensor]: Padded source and target sequences.
    """
    src_inst, tgt_inst = list(zip(*insts))
    src_inst = collate_fn(src_inst)
    tgt_inst = collate_fn(tgt_inst)

    return src_inst, tgt_inst


class CNNDataset(torch.utils.data.Dataset):
    """
    Dataset for style classification or other tasks requiring labeled data.

    Args:
        insts (List[List[int]]): Tokenized instances.
        label (List[int]): Corresponding labels.
    """

    def __init__(self, insts, label):
        self.insts = insts
        self.label = label

    def __getitem__(self, index):
        return self.insts[index], self.label[index]

    def __len__(self):
        return len(self.insts)


def SCIterator(insts_0, insts_1, opt, pad_token_id=1, shuffle=True):
    """
    Creates a data iterator for style classification.

    Args:
        insts_0 (List[Tuple[List[int], int]]): Instances for class 0.
        insts_1 (List[Tuple[List[int], int]]): Instances for class 1.
        opt: Training options, including batch size.
        pad_token_id (int): Padding token ID.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A PyTorch DataLoader for the classification task.
    """
    def cls_fn(insts):
        insts, labels = list(zip(*insts))
        seq = collate_fn(insts, pad_token_id)
        labels = torch.LongTensor(labels)
        return (seq, labels)

    num = len(insts_0) + len(insts_1)
    loader = torch.utils.data.DataLoader(
        CNNDataset(
            insts=insts_0 + insts_1,
            label=[0 if i < len(insts_0) else 1 for i in range(num)]),
        shuffle=shuffle,
        num_workers=2,
        collate_fn=cls_fn,
        batch_size=opt.batch_size)

    return loader


def load_embedding(tokenizer, embed_dim, embed_path=None):
    """
    Loads pre-trained embeddings or generates random embeddings.

    Args:
        tokenizer: Tokenizer object with vocabulary size.
        embed_dim (int): Embedding dimension.
        embed_path (str, optional): Path to pre-trained embeddings.

    Returns:
        np.ndarray: Embedding matrix.
    """
    embedding = np.random.normal(scale=embed_dim ** -0.5, size=(len(tokenizer), embed_dim))
    if embed_path is None:
        return embedding

    print('[Info] Loading embedding')
    embed_dict = {}
    with open(embed_path) as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            tokens = line.rstrip().split()
            try:
                embed_dict[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue

    for i in range(len(tokenizer)):
        try:
            word = tokenizer.decode(i)
            if word in embed_dict:
                embedding[i] = embed_dict[word]
        except:
            print(i)

    return embedding


class BartDataset(torch.utils.data.Dataset):
    """
    Dataset for fine-tuning BART with source and target sequences.

    Args:
        src_inst (List[List[int]]): Source sequences.
        tgt_inst (List[List[int]]): Target sequences.
    """

    def __init__(self, src_inst=None, tgt_inst=None):
        self._src_inst = src_inst
        self._tgt_inst = tgt_inst

    def __len__(self):
        return len(self._src_inst)

    def __getitem__(self, idx):
        return self._src_inst[idx], self._tgt_inst[idx]


def BARTIterator(train_src, train_tgt, valid_src, valid_tgt):
    """
    Creates data iterators for fine-tuning BART.

    Args:
        train_src (List[List[int]]): Training source sequences.
        train_tgt (List[List[int]]): Training target sequences.
        valid_src (List[List[int]]): Validation source sequences.
        valid_tgt (List[List[int]]): Validation target sequences.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for training and validation.
    """
    train_loader = torch.utils.data.DataLoader(
        BartDataset(src_inst=train_src, tgt_inst=train_tgt),
        num_workers=2,
        batch_size=1,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        BartDataset(src_inst=valid_src, tgt_inst=valid_tgt),
        num_workers=2,
        batch_size=1,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader
