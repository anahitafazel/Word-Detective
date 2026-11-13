import random
import pandas as pd
import torch
from torch import nn
from model import TinyWordDetective
SEED = 42
TEST_RATIO = 0.20
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
EPOCHS = 300
LEARNING_RATE = 0.003
DATA_PATH = "word_detective_dataset.csv"
MODEL_PATH = "word_detective_model.pt"
random.seed(SEED)
torch.manual_seed(SEED)
df = pd.read_csv("word_detective_dataset.csv").fillna("")
print(f"Dataset size: {len(df)}")
chars = sorted(set("".join(df["word"].str.lower())))
char_to_id = {
    "<PAD>": 0,
    "<UNK>": 1
}
for char in chars:
    char_to_id[char] = len(char_to_id)
def build_vocab(column):
    values = sorted(
        value for value in df[column].unique()
        if value != ""
    )
    vocab = {
        "<PAD>": 0
    }
    for value in values:
        vocab[value] = len(vocab)
    return vocab
pos_to_id = build_vocab("pos")
past_to_id = build_vocab("past_simple")
pp_to_id = build_vocab("past_participle")
noun_to_id = build_vocab("noun_type")
opposite_to_id = build_vocab("opposite")
max_len = max(len(word) for word in df["word"])
def encode_word(word):
    ids = [
        char_to_id.get(char, char_to_id["<UNK>"])
        for char in word.lower()
    ]
    ids += [char_to_id["<PAD>"]] * (max_len - len(ids))
    return ids
X = torch.tensor(
    [encode_word(word) for word in df["word"]],
    dtype=torch.long
)
def encode_target(value, vocab):
    if value == "":
        return -1
    return vocab[value]
y_pos = torch.tensor(
    [encode_target(v, pos_to_id) for v in df["pos"]],
    dtype=torch.long
)
y_past = torch.tensor(
    [encode_target(v, past_to_id) for v in df["past_simple"]],
    dtype=torch.long
)
y_pp = torch.tensor(
    [encode_target(v, pp_to_id) for v in df["past_participle"]],
    dtype=torch.long
)
y_noun = torch.tensor(
    [encode_target(v, noun_to_id) for v in df["noun_type"]],
    dtype=torch.long
)
y_opposite = torch.tensor(
    [encode_target(v, opposite_to_id) for v in df["opposite"]],
    dtype=torch.long
)
indices = list(range(len(df)))
random.shuffle(indices)
test_size = int(len(indices) * TEST_RATIO)
test_indices = indices[:test_size]
train_indices = indices[test_size:]
print(f"Train size: {len(train_indices)}")
print(f"Test size: {len(test_indices)}")
model = TinyWordDetective(
    vocab_size=len(char_to_id),
    pos_size=len(pos_to_id),
    past_size=len(past_to_id),
    pp_size=len(pp_to_id),
    noun_size=len(noun_to_id),
    opposite_size=len(opposite_to_id),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)
parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)
print(f"Model parameters: {parameter_count}")
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
loss_function = nn.CrossEntropyLoss(
    ignore_index=-1
)
for epoch in range(EPOCHS):
    model.train()
    outputs = model(X[train_indices])
    loss = (
        loss_function(outputs["pos"], y_pos[train_indices])
        + loss_function(outputs["past_simple"], y_past[train_indices])
        + loss_function(outputs["past_participle"], y_pp[train_indices])
        + loss_function(outputs["noun_type"], y_noun[train_indices])
        + loss_function(outputs["opposite"], y_opposite[train_indices])
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()
with torch.no_grad():
    outputs = model(X[test_indices])
    predictions = torch.argmax(
        outputs["pos"],
        dim=1
    )
    accuracy = (
        predictions == y_pos[test_indices]
    ).float().mean().item() * 100
print(f"Test POS Accuracy: {accuracy:.2f}%")
print("Training final model on full dataset...")
model = TinyWordDetective(
    vocab_size=len(char_to_id),
    pos_size=len(pos_to_id),
    past_size=len(past_to_id),
    pp_size=len(pp_to_id),
    noun_size=len(noun_to_id),
    opposite_size=len(opposite_to_id),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
for epoch in range(EPOCHS):
    model.train()
    outputs = model(X)
    loss = (
        loss_function(outputs["pos"], y_pos)
        + loss_function(outputs["past_simple"], y_past)
        + loss_function(outputs["past_participle"], y_pp)
        + loss_function(outputs["noun_type"], y_noun)
        + loss_function(outputs["opposite"], y_opposite)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
torch.save(
    {
        "model_state": model.state_dict(),
        "char_to_id": char_to_id,
        "pos_to_id": pos_to_id,
        "past_to_id": past_to_id,
        "pp_to_id": pp_to_id,
        "noun_to_id": noun_to_id,
        "opposite_to_id": opposite_to_id,
        "max_len": max_len,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "test_accuracy": accuracy
    },
    MODEL_PATH
)
print(f"Final model saved to: {"word_detective_model.pt"   }")