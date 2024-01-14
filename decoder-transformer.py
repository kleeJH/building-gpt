
# Reference: Andrej Karpathy - https://www.youtube.com/@AndrejKarpathy/featured
import torch
import torch.nn as nn
from torch.nn import functional as F

###################################################################################
# Hyper Params
batch_size = 32 # How many independent sequences will we process in parallel?
block_size = 8 # Input chunk size
epoch = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
n_heads = 4
n_layer = 1 # number of transformer blocks
dropout = 0.1
###################################################################################

torch.manual_seed(101)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('./input/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create Encoder and Decoder using character-level tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create dictionaries to map characters to index, vice versa
str_to_idx = { ch:i for i,ch in enumerate(chars) }
idx_to_str = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [str_to_idx[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([idx_to_str[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Batch data generation for x and y
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # disable gradient calculation, reduce memory consumption, used when you do not need to use Tensor.backward()
def estimate_loss():
    # Averaging out the loss for eval_iters samples, using both train and valid dataset
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class SelfAttentionHead(nn.Module):
    """
    Self-Attention head.
    Scaled Dot-Product Attention from Attention Is All You Need (https://arxiv.org/abs/1706.03762)
    """
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # not a parameter and need to assign to pytorch module using register_buffer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Computation of attention scores using the formula from Attention Is All You Need
    
        B, T, C = x.shape # C is the head_size
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores
        weights = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) = (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)

        # aggregate weights
        v = self.value(x) # (B, T, C)
        res = weights @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return res
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.
    """

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # each self-attention block provides (B, T , n_embed // n_attention)
        # after cat n_attention self-attention, it provides (B, T , n_embed)
        res = torch.cat([attention(x) for attention in self.heads], dim=-1)
        res = self.projection(res) # projection layer going back into residual pathway
        res = self.dropout(res)
        return res
    
class FeedForward(nn.Module):
    """
    Linear layer with an activation function.

    Check "Position-wise Feed-Forward Network" in Attention is All You Need

    Note:
        - Self-attention is the communication between tokens, and after the data is gathered
        - This FeedForward layer is used to think upon the gathered data individually
    """

    def __init__(self, n_embed) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # projection layer going back into residual pathway
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x) # (B,T,C)
    
class TransformerBlock(nn.Module):
    """
    Decoder-only transformer block.
    """

    def __init__(self, num_embed, num_heads) -> None:
        super().__init__()
        head_size = num_embed // num_heads
        self.multi_attention = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForward(num_embed)
        self.layer_norm_1 = nn.LayerNorm(num_embed)
        self.layer_norm_2 = nn.LayerNorm(num_embed)

    def forward(self, x):
        x = x + self.multi_attention(self.layer_norm_1(x)) # (B,T,C), apply multiple self-attention heads
        x = x + self.ff(self.layer_norm_2(x)) # (B,T,C)
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # an encoded character will take it's corresponding row from the embedding table
        # i.e. each token directly reads off the logits for the next token from a lookup table (embedding table)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # 4 heads of 8-dim self-attention, after concat, it will return with (B, T , n_embed or C) 
        # *[...transformer blocks], * helps with unpacking the array
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(n_embed, n_heads) for _ in (n_layer)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.linear = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) / (batch, block_size) tensor of integers 
        tkn_embeds = self.token_embedding_table(idx) # (B,T,C) (batch, block_size, n_embed)
        pos_embds = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tkn_embeds + pos_embds # (B,T,C) + (T,C) = (B,T,C) # pytorch will expand another batch dimension for pos_embeds
        x = self.transformer_blocks(x) # (B,T,C)
        x = self.layer_norm(x)
        logits = self.linear(x) # (batch, block_size, vocab_size)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # reshape logits and target because F.cross_entropy requires the shape of (B, C, T)
            # squeeze batches into the same dimension
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx such that it only has block_size number of elements in each batch
            # we need this so the embeddings fit the positional embedding
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_crop)
            # focus only on the last time step (block_size)
            # that means, history of previous predictions (excluding the last prediction) are not used
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(epoch):
    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Epoch #{iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Training
    logits, loss = model(xb, yb) # evaulate loss
    optimizer.zero_grad(set_to_none=True) # zeroing out gradients from the prev step 
    loss.backward() # do backpropagation, get gradients from all the parameters
    optimizer.step() # use backpropagation gradients to update parameters using optimization algorithm

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # generate 500 tokens