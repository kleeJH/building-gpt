
# Reference: Andrej Karpathy - https://www.youtube.com/@AndrejKarpathy/featured
import torch
import torch.nn as nn
from torch.nn import functional as F

###################################################################################
# Hyper Params
batch_size = 32 # How many independent sequences will we process in parallel?
block_size = 8 # Input chunk size
epoch = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # an encoded character will take it's corresponding row from the embedding table
        # i.e. each token directly reads off the logits for the next token from a lookup table (embedding table)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) / (batch, block_size) tensor of integers 
        logits = self.token_embedding_table(idx) # (B,T,C) (batch, block_size, vocab_size)

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
            # get the predictions
            logits, loss = self(idx)
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

model = BigramLanguageModel(vocab_size)
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