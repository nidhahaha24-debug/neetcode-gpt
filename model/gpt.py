
import torch
import torch.nn as nn

class GPT(nn.Module):

    def __init__(self, vocab_size, context_length, model_dim, num_blocks, num_heads):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, model_dim)
        self.position_embeddings = nn.Embedding(context_length, model_dim)

        self.transformer_blocks = nn.Sequential()

        for _ in range(num_blocks):
            self.transformer_blocks.append(
                self.TransformerBlock(model_dim, num_heads)
            )

        self.final_norm = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context):
        embedded = self.word_embeddings(context)

        positions = torch.arange(context.shape[1], device=context.device)
        embedded = embedded + self.position_embeddings(positions)

        output = self.final_norm(self.transformer_blocks(embedded))
        logits = self.vocab_projection(output)

        return logits

    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):

                def __init__(self, model_dim, head_size):
                    super().__init__()

                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)

                def forward(self, embedded):
                    k = self.key_gen(embedded)
                    q = self.query_gen(embedded)
                    v = self.value_gen(embedded)

                    scores = q @ k.transpose(1, 2)
                    scores = scores / (k.shape[-1] ** 0.5)

                    T = embedded.shape[1]
                    mask = torch.tril(
                        torch.ones(T, T, device=embedded.device)
                    ) == 0

                    scores = scores.masked_fill(mask, float('-inf'))
                    weights = torch.softmax(scores, dim=-1)

                    return weights @ v

            def __init__(self, model_dim, num_heads):
                super().__init__()

                assert model_dim % num_heads == 0

                head_size = model_dim // num_heads

                self.att_heads = nn.ModuleList([
                    self.SingleHeadAttention(model_dim, head_size)
                    for _ in range(num_heads)
                ])

                self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

            def forward(self, embedded):
                outputs = [head(embedded) for head in self.att_heads]
                concatenated = torch.cat(outputs, dim=-1)
                return self.output_proj(concatenated)

        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim):
                super().__init__()

                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                return self.dropout(
                    self.down_projection(self.relu(self.up_projection(x)))
                )

        def __init__(self, model_dim, num_heads):
            super().__init__()

            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.linear_network = self.VanillaNeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded):
            embedded = embedded + self.attention(self.first_norm(embedded))
            embedded = embedded + self.linear_network(self.second_norm(embedded))
            return embedded
