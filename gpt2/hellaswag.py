"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os 
import json
import requests 
import tiktoken 
from tqdm import tqdm
import torch 
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__),"hellaswag")
enc = tiktoken.get_encoding("gpt2")
def download_file(url:str, fname:str, chunksize=1024):
    """Helper Function to download the file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length",0))

    with open(fname, "wb") as file, tqdm(desc=fname, total = total, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
        for data in resp.iter_content(chunk_size = chunksize):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

def download(split):
    """Downloads Hellaswag data to cache"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}")
        download_file(data_url, data_filename)

def render_example(example):
    """Given a dictionary render as 3 tensors

    -   tokens (the tokens of context + completion of size 4 x N, as there are only 4 candidates)
    -   mask (is 1 in the region of the candidate completion where we evaluate the likihoods)
    -   laebl the index of the correct completion
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label":label,
        "ctx_tokens":None,
        "ending_tokens":[]
    }

    # gather up all of the tokens 
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows =[]
    for end in endings:
        end_tokens = enc.encode(" "+end) # prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)
    # hae to be careful during the collation becuase the number of tokens iin each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4,max_len), dtype=torch.long)
    mask = torch.zeros((4,max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    return data, tokens, mask, label

def iterate_examples(split):
    #there are 10,042 examples total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

def evaluate(model_type, device):
    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get logits
        logits = model(tokens).logits

        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[...,:-1, :]).contiguous()
        shift_tokens = (tokens[...,1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)

        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now we get the avverage loss just for the sompletion region where mask ==1 in each row 
        shift_mask = mask[...,1:].contiguous() # shift the mask so we start at the last prompt token 
        masked_shift_losses = shift_losses * shift_mask
        #sum and diide by the numb of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        print(sum_loss, avg_loss)
        #now we have the loss for each of the 4 completions
        #The one with the lowest loss is the label 
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()
        # accumulate the stats
        num_total+=1
        num_correct +=int(pred==label)
        num_correct_norm += int(label == pred_norm)
        print(f"{num_total} acc_norm: {num_correct_norm} / {num_total} = {num_correct_norm / num_total:.4f}")

if __name__=="__main__":
    import argparse
    parser  = argparse.ArgumentParser()
    parser.add_argument('-m', "--model_type", type=str, default="gpt2", help="model type to use")
    parser.add_argument("-d", "--device", type=str, default = "cuda", help="device to use")

    args = parser.parse_args()
    evaluate(args.model_type, args.device)