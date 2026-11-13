from pathlib import Path
import torch
from model import TinyWordDetective
BASE = Path(__file__).parent
MODEL_PATH = BASE / "word_detective_model.pt"
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)
model = TinyWordDetective(
    vocab_size=len(
        checkpoint["char_to_id"]
    ),
    pos_size=len(
        checkpoint["pos_to_id"]
    ),
    past_size=len(
        checkpoint["past_to_id"]
    ),
    pp_size=len(
        checkpoint["pp_to_id"]
    ),
    noun_size=len(
        checkpoint["noun_to_id"]
    ),
    opposite_size=len(
        checkpoint["opposite_to_id"]
    ),
    embedding_dim=checkpoint[
        "embedding_dim"
    ],
    hidden_dim=checkpoint[
        "hidden_dim"
    ]
)
model.load_state_dict(
    checkpoint["model_state"]
)
model.eval()
def encode(word):
    char_to_id = checkpoint[
        "char_to_id"
    ]
    ids = [
        char_to_id.get(char, 1)
        for char in word.lower()
    ]
    ids = ids[
        :checkpoint["max_len"]
    ]
    ids += [
        0
    ] * (
        checkpoint["max_len"]
        - len(ids)
    )
    return torch.tensor(
        [ids],
        dtype=torch.long
    )
def reverse_vocab(vocab):
    return {
        index: value
        for value, index in vocab.items()
    }
pos_vocab = reverse_vocab(
    checkpoint["pos_to_id"]
)
past_vocab = reverse_vocab(
    checkpoint["past_to_id"]
)
pp_vocab = reverse_vocab(
    checkpoint["pp_to_id"]
)
noun_vocab = reverse_vocab(
    checkpoint["noun_to_id"]
)
opposite_vocab = reverse_vocab(
    checkpoint["opposite_to_id"]
)
def predict(word):
    with torch.no_grad():
        outputs = model(
            encode(word)
        )
    pos = pos_vocab[
        outputs["pos"]
        .argmax(1)
        .item()
    ]
    print()
    print(f"Word: {word}")
    print(f"Part of Speech: {pos}")
    if pos == "Verb":
        past = past_vocab[
            outputs["past_simple"]
            .argmax(1)
            .item()
        ]
        pp = pp_vocab[
            outputs["past_participle"]
            .argmax(1)
            .item()
        ]
        print(
            f"Past Simple (P): {past}"
        )
        print(
            f"Past Participle (PP): {pp}"
        )
    elif pos == "Noun":
        noun_type = noun_vocab[
            outputs["noun_type"]
            .argmax(1)
            .item()
        ]
        print(
            f"Type: {noun_type}"
        )
    elif pos == "Adjective":
        opposite = opposite_vocab[
            outputs["opposite"]
            .argmax(1)
            .item()
        ]
        print(
            f"Opposite: {opposite}"
        )
print("Word Detective")
print("Type an English word.")
print("Type 'exit' to stop.")
while True:
    word = input("\nWord: ").strip()
    if word.lower() == "exit":
        break
    if word:
        predict(word)