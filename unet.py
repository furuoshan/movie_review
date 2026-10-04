import torch
import torch.nn as nn
from math import sin, cos


class ReviewClassifier(nn.Module):
    def __init__(self, embedding_dim, max_length, dataloader, vocab=None):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            batch_first=True
            ,dim_feedforward=2048,
        )
        self.linear = nn.Linear(embedding_dim, 1)
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.encoder = nn.TransformerEncoder(layer, num_layers=6, enable_nested_tensor=False)
        self.classifier = nn.Linear(embedding_dim, 2)
        self.vocab = {"<pad>": 0, "<unk>": 1, "<mask>":2} if vocab is None else vocab
        for data in dataloader:
            for text in data["text"]:
                for word in text.lower().split():
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=self.embedding_dim, padding_idx=self.vocab["<pad>"])


    def sinusoidalPositionEmbedding(self, x):
        y = torch.zeros([1, self.max_length, self.embedding_dim], device=x.device)
        for i in range(self.max_length):
            y[0][i] = self.sinusoidal(i)
        return x+y

    def sinusoidal(self, pos):
        v = [[sin(pos / (10000 ** (2 * i / self.embedding_dim))) for i in range(self.embedding_dim//2)],
             [cos(pos / (10000 ** ((2 * i) / self.embedding_dim))) for i in range(self.embedding_dim//2)]]
        v = torch.tensor(v, dtype=torch.float32)
        return v.transpose(0, 1).reshape(-1)

    def forward(self, batch, task=0):
        if task == 0:
            # padding_mask: [B, L]，True 表示补齐位置
            batch = [[self.vocab.get(word, self.vocab["<unk>"]) for word in text] for text in batch]
            batch = torch.tensor(batch, dtype=torch.long, device=self.embedding.weight.device)
            padding_mask = batch.eq(0)
            encoded = self.encoder(
                self.sinusoidalPositionEmbedding(self.embedding(batch)),
                src_key_padding_mask=padding_mask
            )

            # 只对真实词的位置取平均，排除补齐位置
            valid = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * valid).sum(dim=1)
            pooled = pooled / valid.sum(dim=1).clamp(min=1)
            return self.classifier(pooled)
        else:
            # padding_mask: [B, L]，True 表示补齐位置
            batch = [[self.vocab.get(word, self.vocab["<unk>"]) for word in text] for text in batch]
            batch = torch.tensor(batch, dtype=torch.long, device=self.embedding.weight.device)
            valid = batch.ne(self.vocab["<pad>"])
            predict_mask = (torch.rand(batch.shape, device=batch.device) < 0.15) & valid

            encoded = self.encoder(
                self.sinusoidalPositionEmbedding(self.embedding(batch)),
                src_key_padding_mask=predict_mask
            )
            return self.linear()

