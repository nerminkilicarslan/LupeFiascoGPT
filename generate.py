import torch
from model import GPTLanguageModel

device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


def load_model(path="checkpoints/model.pt"):
    ckpt = torch.load(path, map_location=device)
    model = GPTLanguageModel(
        vocab_size=ckpt['vocab_size'],
        n_embd=ckpt['n_embd'],
        block_size=ckpt['block_size'],
        n_head=ckpt['n_head'],
        n_layer=ckpt['n_layer'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['stoi'], ckpt['itos']


def generate_text(model, stoi, itos, prompt, max_tokens=300, temperature=0.8, top_k=50):
    context = torch.tensor(
        [[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device
    )
    output = model.generate(context, max_tokens, temperature=temperature, top_k=top_k)
    return ''.join([itos.get(i, '') for i in output[0].tolist()])


if __name__ == "__main__":
    print("Model yukleniyor...")
    model, stoi, itos = load_model()
    print("Lupe Fiasco GPT | cikmak icin 'q'\n")

    while True:
        prompt = input("Baslangic satiri: ").strip()
        if prompt.lower() == 'q':
            break
        if not prompt:
            prompt = "I"
        output = generate_text(model, stoi, itos, prompt)
        print("\n" + "-" * 48)
        print(output)
        print("-" * 48 + "\n")
