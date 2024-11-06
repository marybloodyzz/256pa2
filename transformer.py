import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class FeedForwardClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(FeedForwardClassifier, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
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

    def forward(self, x, mask=None):
        batch_size, seq_length, n_embd = x.size()
        # kqv pairs
        K = self.k_linear(x)
        Q = self.q_linear(x)
        V = self.v_linear(x)

        K = K.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        Q = Q.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)

        # dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attn_weights = Q @ K.transpose(-2,-1) * n_embd**-0.5

        # print("origin attn_weights size: ", attn_weights.shape)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            # print("maskeed attn_weights size: ", attn_weights.shape)

        attn_probs = F.softmax(attn_weights, dim=-1)
        # print(attn_probs.shape)
        #select one head
        attn_probs = attn_probs[:, 1, :, :].unsqueeze(1)
        # print(attn_probs.shape)
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, n_embd)

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
        self.pos_embd = nn.Embedding(max_context_length, n_embd)
        self.layers = nn.ModuleList(
            [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.n_head = n_head

    def forward(self, x):
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_length)
        x_embd = self.token_embd(x) + self.pos_embd(positions)

        # get a attention mask
        mask = (x != 0).unsqueeze(1).unsqueeze(2)  # the shape should be (batch_size, 1, 1, seq_length)

        attn_maps = []
        x = x_embd
        for layer in self.layers:
            x, attn_probs = layer(x, mask)
            # reduced_attn_map = attn_map.sum(dim=2)  # Resulting shape will be [1, 1, 32, 32]

            for attn_map in attn_probs:
                attn_maps.append(attn_map)
        return x, attn_maps

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super(TransformerDecoderBlock, self).__init__()
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, hidden_dim=100)  # Hidden dimensionality of 100
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        attn_output, attn_probs = self.attn(x, mask)
        x = x + attn_output
        x = self.ln1(x)
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.ln2(x)
        return x, attn_probs

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_context_length):
        super(TransformerDecoder, self).__init__()
        self.token_embd = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.pos_embd = nn.Embedding(max_context_length, n_embd)
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.n_head = n_head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # Output layer

    def forward(self, x, targets=None, return_attn=False):
        
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_length)
        x_embd = self.token_embd(x) + self.pos_embd(positions)
        batch_size, seq_length, _ = x_embd.size()

        # use causal mask
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, self.n_head, seq_length, seq_length)

        x = x_embd
        attn_maps = []
        for layer in self.layers:
            x, attn_probs = layer(x, mask)
            
            for attn_map in attn_probs:
                attn_maps.append(attn_map)

        x = self.ln_f(x)
        logits = self.head(x)
        
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # calculate loss wth ignoring padding index 0
            loss = F.cross_entropy(logits, targets)

            return loss
        else:
            return logits, attn_maps



class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_context_length, n_hidden, n_output):
        super(TransformerClassifier, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, n_embd, n_head, n_layer, max_context_length)
        self.classifier = FeedForwardClassifier(n_embd, n_hidden, n_output)
    
    def forward(self, x):
        encoder_output, _ = self.encoder(x)
        # apply mask for valid tokens and apply it to embeddings
        mask = (x != 0).unsqueeze(-1).float()
        masked_output = encoder_output * mask
        # sum embeddings and divide by valid token counts
        sum_embeddings = masked_output.sum(dim=1)  
        valid_token_counts = mask.sum(dim=1)
        output = sum_embeddings / valid_token_counts
        logits = self.classifier(output)
        return logits