import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_alibi_slopes(n_head):
    def get_slopes(n):
        def power(a, b):
            return a ** b
        start = power(2, -2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]
    if math.log2(n_head).is_integer():
        slopes = get_slopes(n_head)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
        slopes = get_slopes(closest_power_of_2)
        extra_slopes = get_slopes(2 * closest_power_of_2)[0::2]
        slopes.extend(extra_slopes[: n_head - closest_power_of_2])
    return slopes

class FeedForwardClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(FeedForwardClassifier, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd, hidden_dim=256):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_embd)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(n_embd, n_embd)
        self.k_linear = nn.Linear(n_embd, n_embd)
        self.v_linear = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)

        # get AliBi slopes for each head
        slopes = get_alibi_slopes(n_head)
        self.register_buffer('alibi_slopes', torch.tensor(slopes).unsqueeze(1).unsqueeze(1))

    def forward(self, x, mask=None):
        batch_size, seq_length, n_embd = x.size()
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        Q = Q.view(batch_size, self.n_head, seq_length, self.head_dim)
        K = K.view(batch_size, self.n_head, seq_length, self.head_dim)
        V = V.view(batch_size, self.n_head, seq_length, self.head_dim)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, n_head, seq_length, seq_length)
        positions = torch.arange(seq_length, device=x.device)
        relative_positions = positions[None, :] - positions[:, None]
        relative_positions = relative_positions.abs().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, seq_length)

        # apply AliBi biases
        alibi_biases = self.alibi_slopes * relative_positions  # (n_head, 1, seq_length, seq_length)
        attn_weights = attn_weights - alibi_biases

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        #select one head
        attn_probs = attn_probs[:, 1, :, :].unsqueeze(1)
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.view(batch_size, seq_length, n_embd)

        output = self.out_proj(attn_output)

        return output, attn_probs

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        attn_output, attn_probs = self.attn(x, mask)
        x = x + attn_output
        x = self.ln1(x)
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.ln2(x)
        return x, attn_probs

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_context_length):
        super(TransformerEncoder, self).__init__()
        self.token_embd = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.layers = nn.ModuleList(
            [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        )

    def forward(self, x):
        x_embd = self.token_embd(x)
        mask = (x != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)
        attn_maps = []
        x = x_embd
        for layer in self.layers:
            x, attn_probs = layer(x, mask)
            for attn_map in attn_probs:
                attn_maps.append(attn_map)
        return x, attn_maps

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_context_length, n_hidden, n_output):
        super(TransformerClassifier, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, n_embd, n_head, n_layer, max_context_length)
        self.classifier = FeedForwardClassifier(n_embd, n_hidden, n_output)
    
    def forward(self, x):
        encoder_output, attn_maps = self.encoder(x)
        # apply mask for valid tokens and apply embeddings
        mask = (x != 0).unsqueeze(-1).float()
        masked_output = encoder_output * mask
        sum_embeddings = masked_output.sum(dim=1)  
        valid_token_counts = mask.sum(dim=1)
        output = sum_embeddings / valid_token_counts
        logits = self.classifier(output)
        return logits