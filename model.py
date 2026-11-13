import torch
from torch import nn
class TinyWordDetective(nn.Module):
    def __init__(
        self,
        vocab_size,
        pos_size,
        past_size,
        pp_size,
        noun_size,
        opposite_size,
        embedding_dim=32,
        hidden_dim=64
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.pos_head = nn.Linear(hidden_dim, pos_size)
        self.past_head = nn.Linear(hidden_dim, past_size)
        self.pp_head = nn.Linear(hidden_dim, pp_size)
        self.noun_head = nn.Linear(hidden_dim, noun_size)
        self.opposite_head = nn.Linear(hidden_dim, opposite_size)
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        h = hidden[-1]
        return {
            "pos": self.pos_head(h),
            "past_simple": self.past_head(h),
            "past_participle": self.pp_head(h),
            "noun_type": self.noun_head(h),
            "opposite": self.opposite_head(h)
        }