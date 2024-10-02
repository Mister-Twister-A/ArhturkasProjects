import torch
import torch.nn as nn
from torch.nn import functional as F

Train_data = open("tiny-shakespeare.txt", "r")
torch.manual_seed(1337)
text = Train_data.read()
vocab = sorted(list(set(text)))

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
batch_size = 10
block_length = 10
n_emb = 100
n_heads = 4
n_layers = 4
dropout = 0.2

stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for i,s in enumerate(vocab)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda s: ''.join([itos[i] for i in s])

data = torch.tensor(encode(text), dtype=torch.long)
split = int(len(data) * 0.9)
train_set = data[:split]
val_set = data[split:]

def get_batch(split):
    cur_data = train_set if split == "train" else val_set
    idx = torch.randint(len(data) - block_length, (batch_size, ))
    x = torch.stack([data[i:i+block_length] for i in idx])
    y = torch.stack([data[i+1: i+block_length+1] for i in idx])
    x,y = x.to(device),y.to(device)
    return x,y

class Block(nn.Module):

    def __init__(self, num_heads, n_emb):
        super().__init__()
        self.sa_layer = MultiHeadAttention(num_heads,n_emb//num_heads)
        self.fdfr = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    
    def forward(self,x):
        x = x + self.sa_layer(self.ln1(x))
        x = x + self.fdfr(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj_layer = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj_layer(out))
        return out

class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(4 * n_emb,n_emb),
            nn.Dropout(dropout)

        )
    
    def forward(self, x):
        return self.net(x)


class BigramModel(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_len, n_emb)
        self.pos_embeddings = nn.Embedding(block_length, n_emb)
        self.ln_head = nn.Linear(n_emb, vocab_len)
        # self.sa_head = MultiHeadAttention(4,n_emb//4)
        # self.fdfr = FeedForward(n_emb)
        self.blocks = nn.Sequential(*[Block(n_heads, n_emb) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_emb)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tmp_emb = self.embeddings(idx) # (B,T,C)
        pos_emb = self.pos_embeddings(torch.arange(T, device=device)) # (T,C)
        tmp = tmp_emb + pos_emb # (B,T,C)
        tmp = self.blocks(tmp)
        tmp = self.ln(tmp)
        # tmp = self.sa_head(tmp)
        # tmp = self.fdfr(tmp)
        logits = self.ln_head(tmp) # (B,T,vocab_len)
        #print(logits.shape, "yaya")
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            #print(logits.shape)
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) of current conext (B - batches, T - block_lenght)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_length:]
            logits,loss = self(idx_cond)
            logits = logits[:, -1, :] # current context
            #print(logits.shape)
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,pred), dim=1)
        return idx


class Head(nn.Module): # sigle step of self attention
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.key_layer = nn.Linear(n_emb, num_heads, bias=False)
        self.query_layer = nn.Linear(n_emb, num_heads, bias=False)
        self.value_leayer = nn.Linear(n_emb, num_heads, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_length, block_length)))

    def forward(self, x):
        B,T,C = x.shape
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_leayer(x)
        wei = query @ key.transpose(-2,-1) * self.num_heads**-0.5
        wei = self.dropout(wei)

        #tril = torch.tril(torch.ones(T,T))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ value

        return out

m = BigramModel(len(vocab))
m = m.to(device)

optimize = torch.optim.AdamW(m.parameters(), lr=3e-4)
for steps in range(5001):
    xs,ys = get_batch("train")

    logits,loss = m(xs,ys)
    optimize.zero_grad(set_to_none=True)
    loss.backward()
    optimize.step()
    if steps % 1000 == 0:
        print(f"step {steps} loss {loss.item()}")





xs,ys = get_batch("train")
lg,l = m(xs,ys)
test = xs
ans = m.generate(test, 1000)
fun_test = torch.tensor(encode("yay eight times"),device=device)
fun_test = fun_test.view(1,-1)
print(fun_test, fun_test.shape)
fun_ans =m.generate(fun_test, 1000)
print(ans)
print(decode(fun_ans[0].tolist()))


