import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples


class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # Key, Value, Query for all heads but in one batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embed

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence_len, embedding_dim n_embed
        # calculate the query, key and value for all heads in a batch and move forward to be the batch
        # nh is number of heads, hs is head size and c is number of channels
        # e.g. in GPT-2 (124M), n_heads = 12, hs=64, sonh*hs =C=768 in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B,nh, T, ns
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B,nh, T, ns
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B,nh, T, ns
        # Attenion (materialisze the large T,T matrix for all the queryies and keys )
        # This is flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # Get back into the shape it came in
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges = 256 byte tokens + 1 <|endoftext|>
    )
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.block_size, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is always B,T
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is {config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # positional embeddings of shape T,N-embed
        tok_emb = self.transformer.wte(idx)  # token embedding of shape B,T,n_embed
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward through final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # B,T,Vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained model GPT-2 Model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained gpt: { model_type}")
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always this for GPT model checkpoints
        config_args["block_size"] = (
            1024  # always this sequence length for GPT chekpoints
        )
        # create a sfrom scratch intialised minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer

        # init a huggingface transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # copy while ensuring all of the parameters are aligned and match in name and shapes
        sd_hf_keys = sd_hf.keys()
        sd_hf_keys = [
            k for k in sd_hf_keys if not k.endswith(".attn.masked_bias")
        ]  # discard this mask / buffer
        sd_hf_keys = [
            k for k in sd_hf_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        assert len(sd_keys) == len(
            sd_hf_keys
        ), f"Mismatched length of keys {len(sd_keys)} != len(sd_hf_keys)"

        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                # special treatemnet for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimiser(self, weight_decay, lr, device):
        # start with all of the candidate parameters that require_grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups, any params that is 2d will be weight decayed else we ignore
        # i.e. all weight tensors in matmuls + embeddings decay and al biases and layer norms wont
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        if master_process:
            print(f"Num of decay_params = {num_decay_params}")
            print(f"Num of no_decay_params = {num_no_decay_params}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        if master_process:
            print(f"Using fused adam {use_fused}")
        optimiser = torch.optim.AdamW(
            optim_groups, lr=lr, betas=(0.9, 0.96), eps=1e-8, fused=use_fused
        )
        return optimiser


import tiktoken
import numpy as np


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataloaderShardLite:

    def __init__(self, B, T, process_rank, num_processes, split):

        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ["train", "val"]

        data_root = "edu_fineweb-10BT"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"We found no shards for split {split}"
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        # if loading next batch would be out of bounds we rest
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()

    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now we get the avverage loss just for the sompletion region where mask ==1 in each row
    shift_mask = mask[
        ..., 1:
    ].contiguous()  # shift the mask so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and diide by the numb of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    pred_norm = avg_loss.argmin().item()
    return pred_norm


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP distributed data parallel
# torch run sets the env RANk, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    # use DDP atm demands CUDA we set the device appropriately according to rank
    assert torch.cuda.is_available(), "We need cuda for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # num of GPUS

    torch.cuda.set_device(ddp_local_rank)  # changed this to set device for DDP
    # torch.cuda.set_device(device)
    device = f"cuda:{ddp_local_rank}"

    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla non DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1  # num of GPUS
    master_process = True
    # attempt to auto detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

batch_size = 8
context_len = 1024
total_batch_size = 524288  # 2**19, ~0.5 tokens

assert (
    total_batch_size % (batch_size * context_len * ddp_world_size) == 0
), "Make sure total batch size is divisble by the batch and context length 8 ddp_world_size"
grad_accum_steps = total_batch_size // (batch_size * context_len * ddp_world_size)

if master_process:
    print(f"The desired batch size is {total_batch_size}")
    print(f"The calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataloaderShardLite(
    batch_size,
    context_len,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train",
)
val_loader = DataloaderShardLite(
    batch_size,
    context_len,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="val",
)

torch.set_float32_matmul_precision("high")
# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
use_compile = False  # torch compile messes with Hellaswag

if use_compile and device.startswith("cuda"):
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073


def get_lr(it):
    # lienar warmup for warmup iters
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # if it > lr_+decay_, rturn min lr
    if it > max_steps:
        return min_lr
    # in between use cosine decay down to min
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, betas= (0.9, 0.95), eps = 1e-8)
optimiser = raw_model.configure_optimiser(weight_decay=0.1, lr=6e-4, device=device)
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "log.txt")
if master_process:
    with open(log_file, "w") as f:  # clear the log file
        pass
for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1
    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionall write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                torch.save(checkpoint, checkpoint_path)

    # once in a while we gonna evaluate on hellaswag
    if ((step > 0 and step % 250 == 0) or last_step) and not (use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i%ddp_world_size==ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _ = model(tokens)  # B,T,Vocab_size
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(
                num_correct_norm, dtype=torch.long, device=device
            )
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(
                f"Hellaswag accuracy is {num_correct_norm}/{num_total}: {num_correct_norm/num_total}"
            )
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
    # once in a while we generate from the model except step 0, which is noise
    if ((step > 0 and step % 250 == 0) or last_step) and not (use_compile):
        model.eval()
        num_return_sentences = 4
        max_length = 32
        tokens = enc.encode("Hello I am a language model,")
        tokens = torch.tensor(tokens)
        tokens = tokens.unsqueeze(0).repeat(num_return_sentences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # for ward to get the logits
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(xgen)  # B,T,Vocab_size
            logits = logits[:, -1, :]  # B,Vocabsize
            probs = F.softmax(logits, dim=-1)
            # do top k sampling
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabiliesi
            # note multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # B,1
            # gather the correspinding indicies
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_return_sentences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank: {ddp_rank} sample {i}: {decoded}")

    model.train()
    optimiser.zero_grad()

    # gradient accumulation
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation
        # because the gradients just add on each successive backwards pass
        # addition of gradients corresponds with a SUM in the objectie but instead of
        # SUM we want MEAN. Scale the loss here so it comes out as normalised / scaled correctly
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the lr for this iteration
    lr = get_lr(step)
    for param_group in optimiser.param_groups:
        param_group["lr"] = lr
    optimiser.step()
    torch.cuda.synchronize()  # wait for gpus to sync
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    ) / dt
    if master_process:
        print(
            f"step {step} ~ lr: {lr} ~ loss: {loss_accum.item():.2f} ~ norm: {norm} ~ {dt*1000:.2f} ms tokens per second {tokens_per_sec}"
        )
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.4f}\n")
if ddp:
    destroy_process_group()
