from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Parameters in billions: {total_params / 1e9:.2f}B")
def ask_word_detective(word):
    prompt = f"""
You are Word Detective, an English vocabulary assistant.
Analyze the following English word:
Word: {word}
Give exactly these five pieces of information:
1. Part of speech
2. Simple past form
3. Past participle
4. Noun form
5. Opposite
Important rules:
- For "Noun form", give the noun derived from the word itself.
- Do not give synonyms, related words, examples, or words with a similar meaning.
- If there is no common derived noun, write "No common noun form".
- For verb forms, give the correct simple past and past participle.
- If a form does not apply, write "Not applicable".
- For "Opposite", give one common opposite word.
- Keep the answer short and clear.
Example:
Word: choose
Part of speech: verb
Simple past form: chose
Past participle: chosen
Noun form: choice
Opposite: refuse
"""
    messages = [
        {
            "role": "system",
            "content": "You are a precise English vocabulary assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120
        )
    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return result
if __name__ == "__main__":
    word = input("Enter an English word: ")
    answer = ask_word_detective(word)
    print("\nWord Detective:")
    print(answer)