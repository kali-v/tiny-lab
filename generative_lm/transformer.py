
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, context_size, mask=None):
        super().__init__()
        self.in_size = emb_size
        self.num_heads = num_heads
        self.context_size = context_size
        head_size = emb_size // num_heads

        self.q_layers = nn.ModuleList()
        self.k_layers = nn.ModuleList()
        self.v_layers = nn.ModuleList()

        for _ in range(num_heads):
            self.k_layers.append(nn.Linear(emb_size, head_size, bias=False))
            self.q_layers.append(nn.Linear(emb_size, head_size, bias=False))
            self.v_layers.append(nn.Linear(emb_size, head_size, bias=False))

        self.projection = nn.Linear(emb_size, emb_size)
        self.ln = nn.LayerNorm(emb_size)

        self._init_params()

        self.register_buffer('mask', mask)

    def _init_params(self):
        for l in [*self.q_layers, *self.k_layers, *self.v_layers, self.projection]:
            nn.init.xavier_uniform_(l.weight)

    def _sdp_attention(self, q, k, v):
        att_log = (q @ k.transpose(-2, -1))
        if self.mask is not None:
            att_log = att_log.masked_fill(self.mask == 0, float("-inf"))
        attention = F.softmax(att_log, dim=-1)
        return attention @ v

    def forward(self, x):
        values = torch.tensor([], device=x.device)
        for b in range(self.num_heads):
            qkv = self.q_layers[b](x), self.k_layers[b](x), self.v_layers[b](x)
            val = self._sdp_attention(*qkv)
            values = torch.cat((val, values), dim=-1)

        out = self.projection(values) + x
        nout = self.ln(out)

        return nout


class FFN(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_size)
        )
        self.ln = nn.LayerNorm(emb_size)
        self.ffn.apply(self._init_params)

    def _init_params(self, l):
        if type(l) == nn.Linear:
            nn.init.xavier_uniform_(l.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        return self.ln(self.ffn(x) + x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, context_size):
        super().__init__()
        self.context_size = context_size

        mask_tril = torch.tril(torch.ones(context_size, context_size))

        self.block = nn.Sequential(
            MultiHeadAttention(emb_size, num_heads, context_size, mask_tril),
            MultiHeadAttention(emb_size, num_heads, emb_size, mask_tril),
            FFN(emb_size, emb_size * 8),
        )

    def forward(self, x):
        return self.block(x)


class TransformerModel(nn.Module):
    def __init__(self, emb_size, num_blocks, num_heads, context_size, vocab_size):
        super().__init__()
        self.context_size = context_size

        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_embedding = nn.Embedding(context_size, emb_size)

        self.blocks = nn.Sequential(*[
            TransformerBlock(emb_size, num_heads, context_size) for _ in range(num_blocks)
        ])

        self.out = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        te = self.token_embedding(x)
        inp = te + self.pos_embedding(torch.arange(self.context_size).to(te.device))
        blocks_out = self.blocks(inp)
        return self.out(blocks_out)

    @torch.no_grad()
    def generate_tokens(self, x, gen_len, top_k=5):
        x = x[0]

        for _ in range(gen_len):
            pv, pi = self(x)[-1, :].topk(top_k)
            p = F.softmax(pv, dim=-1)
            q = pi[torch.multinomial(p, num_samples=1)]

            x = torch.cat((x, q), dim=-1)

        return x
