import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch import cuda
from torch.nn import CrossEntropyLoss
from model import BartModel, BartForMaskedLM
from transformers import BartTokenizer
from utils.helper import make_padding_mask, optimize, evaluate, cal_sc_loss, cal_bl_loss
from utils.dataset import read_data, BARTIterator
from utils.optim import ScheduledOptim
import pickle
import random
from sklearn.model_selection import train_test_split

# Configure the device for computation (CUDA if available)
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define filter sizes and number of filters for potential CNN-based tasks
filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]

# Custom padding mask creation
def make_padding_mask(input_ids, padding_idx=1):
    """Creates a padding mask for input sequences."""
    return input_ids.eq(padding_idx)

def main():
    """Main function to train a BART-based model with BLEU reward and cross-entropy loss."""
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # Load the BART model and its masked language model counterpart
    base = BartModel.from_pretrained("facebook/bart-base")
    model = BartForMaskedLM.from_pretrained('facebook/bart-base', config=base.config)
    model.to(device).train()  # Move model to the device and set it to training mode

    # Load training and validation data
    with open("traintextCMwithCS.pkl", "rb") as fp:
        traintext = pickle.load(fp)
    with open("trainCMlabelAndSSAndSpan.pkl", "rb") as fp:
        trainlabels = pickle.load(fp)

    for i in range(len(traintext)):
        traintext[i] = str(traintext[i])
    for i in range(len(trainlabels)):
        trainlabels[i] = str(trainlabels[i])

    with open("validtextCMwithCS.pkl", "rb") as fp:
        validtext = pickle.load(fp)
    with open("validCMlabelAndSSAndSpan.pkl", "rb") as fp:
        validlabels = pickle.load(fp)

    for i in range(len(validtext)):
        validtext[i] = str(validtext[i])
    for i in range(len(validlabels)):
        validlabels[i] = str(validlabels[i])

    print(len(validtext))

    # Prepare training and validation sequences
    trainsrc_seq, traintgt_seq, validsrc_seq, validtgt_seq = [], [], [], []
    max_len = 30  # Maximum sequence length

    # Tokenize and process training data
    f1, f2 = traintext, trainlabels
    index = list(range(len(f1)))
    random.shuffle(index)
    for i, (s, t) in enumerate(zip(f1, f2)):
        if i in index:
            if len(s) == 0:
                continue
            s = tokenizer.encode(s)
            t = tokenizer.encode(t)
            s = s[:min(len(s) - 1, max_len)] + s[-1:]
            t = t[:min(len(t) - 1, max_len)] + t[-1:]
            trainsrc_seq.append(s)
            traintgt_seq.append([tokenizer.bos_token_id] + t)

    # Tokenize and process validation data
    f1, f2 = validtext, validlabels
    index = list(range(len(f1)))
    random.shuffle(index)
    for i, (s, t) in enumerate(zip(f1, f2)):
        if i in index:
            s = tokenizer.encode(s)
            t = tokenizer.encode(t)
            s = s[:min(len(s) - 1, max_len)] + s[-1:]
            t = t[:min(len(t) - 1, max_len)] + t[-1:]
            validsrc_seq.append(s)
            validtgt_seq.append([tokenizer.bos_token_id] + t)

    # Create data loaders
    train_loader, valid_loader = BARTIterator(trainsrc_seq, traintgt_seq, validsrc_seq, validtgt_seq)

    print('done')

    # Define loss function and optimizer
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09),
        1e-5, 10000
    )

    # Training loop
    tab = 0
    eval_loss = 1e8  # Initialize evaluation loss
    total_loss_ce, total_loss_co = [], []  # Store loss values
    start = time.time()
    train_iter = iter(train_loader)

    for step in range(1, 30002):
        print('current {}'.format(step))

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        src, tgt = map(lambda x: x.to(device), batch)
        src_mask = make_padding_mask(src, tokenizer.pad_token_id)
        src_mask = 1 - src_mask.long() if src_mask is not None else None
        logits = model(src, attention_mask=src_mask, decoder_input_ids=tgt)[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tgt[..., 1:].contiguous()
        loss_ce = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_loss_ce.append(loss_ce.item())

        # BLEU reward loss
        loss_co = torch.tensor(0)
        if step > 10000 or step > len(train_loader):
            idx = tgt.ne(tokenizer.pad_token_id).sum(-1).to(device)
            loss_co = cal_bl_loss(logits, tgt, idx, tokenizer)
            total_loss_co.append(loss_co.item())

        optimize(optimizer, loss_ce + loss_co)

        # Log training progress
        if step % 100 == 0:
            lr = optimizer._optimizer.param_groups[0]['lr']
            print('[Info] steps {:05d} | loss_ce {:.4f} | loss_co {:.4f} | lr {:.6f} | second {:.2f}'.format(
                step, np.mean(total_loss_ce), np.mean(total_loss_co), lr, time.time() - start))
            total_loss_ce, total_loss_co = [], []
            start = time.time()

        # Save model checkpoints
        if step % 2000 == 0:
            os.makedirs('SS', exist_ok=True)
            torch.save(model.state_dict(), 'SS/{}.chkpt'.format(step))

        # Evaluate model
        if step % 200 == 0 or step % len(train_loader) == 0:
            valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, tokenizer, step)
            if eval_loss >= valid_loss:
                torch.save(model.state_dict(), 'checkpoints/{}.chkpt'.format(step))
                print('[Info] The checkpoint file has been updated.')
                eval_loss = valid_loss
                tab = 0
            else:
                tab += 1
            if tab == 10:  # Early stopping condition
                exit()

if __name__ == "__main__":
    main()
