# Lupe Fiasco GPT

A character-level GPT language model trained on Lupe Fiasco's rap lyrics. Given a starting line, the model generates new lyrics in his style.

---

## How It Works

The model is a transformer-based character-level language model built from scratch with PyTorch. It learns patterns at the character level — rhythm, word choice, line breaks — directly from Lupe Fiasco's discography.

**Architecture**

- 6 transformer blocks
- 8 attention heads
- 256-dimensional embeddings
- 256 character context window
- ~10M parameters

**Training data**: 149 songs pulled from the Genius API, covering Food & Liquor, The Cool, Lasers, Tetsuo & Youth, and Drill Music in Zion.

---

## Setup

**Requirements**

- Python 3.10+
- pip

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Collect lyrics** (requires a free Genius API token from genius.com/api-clients)

```bash
GENIUS_TOKEN=your_token_here python3 collect_lyrics.py
```

**Train the model**

```bash
python3 train.py
```

Training runs for 10,000 steps and saves checkpoints every 500 steps to `checkpoints/model.pt`. On Apple Silicon (MPS) this takes roughly 60-90 minutes.

**Launch the web interface**

```bash
python3 app.py
```

Then open `http://localhost:7860` in your browser.

---

## Usage

The web interface lets you type a starting line or pick one of the preset prompts, adjust three parameters, and generate lyrics.

| Parameter | Description |
|-----------|-------------|
| Creativity (temperature) | Higher = more unexpected, lower = more predictable |
| Length (tokens) | Number of characters to generate |
| Boldness (top-k) | Sampling pool size; higher = more varied word choices |

You can also run the model directly in the terminal:

```bash
python3 generate.py
```

---

## File Structure

```
LupeFiascoGPT/
├── model.py          # GPT architecture (transformer blocks, attention)
├── dataset.py        # Character-level tokenizer and train/val split
├── train.py          # Training loop and hyperparameters
├── generate.py       # CLI inference
├── collect_lyrics.py # Genius API scraper
├── app.py            # Gradio web interface
├── requirements.txt
└── checkpoints/
    └── model.pt      # Saved model weights (not tracked in git)
```

---

## Notes

- The model does not understand meaning — it predicts the next character based on learned patterns. Output is intentionally imperfect and experimental.
- `lupe_dataset.txt` and `checkpoints/model.pt` are excluded from version control. Run the collect and train steps locally to reproduce them.

---

---

# Lupe Fiasco GPT (Turkce)

Lupe Fiasco'nun rap sozlerinden egitilmis karakter duzeyli bir GPT dil modeli. Bir baslangic satiri verildiginde model, onun tarzinda yeni sozler uretir.

---

## Nasil Calisir

Model, PyTorch ile sifirdan yazilmis transformer tabanli bir karakter duzeyli dil modelidir. Ritim, kelime secimi ve satir kirilmalari gibi kaliplari karakter duzeyinde, dogrudan Lupe Fiasco'nun diskografisinden ogrenilir.

**Mimari**

- 6 transformer blok
- 8 dikkat kafasi
- 256 boyutlu gomme katmani
- 256 karakterlik baglam penceresi
- Yaklasik 10 milyon parametre

**Egitim verisi**: Genius API araciligiyla cekilmis 149 sarki. Food & Liquor, The Cool, Lasers, Tetsuo & Youth ve Drill Music in Zion albumlerini kapsar.

---

## Kurulum

**Gereksinimler**

- Python 3.10+
- pip

**Bagimliliklari kur**

```bash
pip install -r requirements.txt
```

**Sarki sozlerini topla** (genius.com/api-clients adresinden ucretsiz bir Genius API tokeni gerekir)

```bash
GENIUS_TOKEN=tokenini_buraya_yaz python3 collect_lyrics.py
```

**Modeli egit**

```bash
python3 train.py
```

Egitim 10.000 adim surar ve her 500 adimda bir `checkpoints/model.pt` dosyasina kaydeder. Apple Silicon (MPS) uzerinde yaklasik 60-90 dakika surer.

**Web arayuzunu baslat**

```bash
python3 app.py
```

Ardindan tarayicinda `http://localhost:7860` adresini ac.

---

## Kullanim

Web arayuzu uzerinden bir baslangic satiri yazabilir ya da hazir seceneklerden birini kullanabilirsin. Uc parametre ayarlanabilir.

| Parametre | Aciklama |
|-----------|----------|
| Creativity (yaraticilik) | Yuksek = daha beklenmedik, dusuk = daha tahmin edilebilir |
| Length (uzunluk) | Uretilecek karakter sayisi |
| Boldness (cesaret) | Ornekleme havuzu; yuksek = daha cesur kelime secimleri |

Modeli dogrudan terminalde de calistiabilirsin:

```bash
python3 generate.py
```

---

## Dosya Yapisi

```
LupeFiascoGPT/
├── model.py          # GPT mimarisi (transformer bloklari, dikkat)
├── dataset.py        # Karakter duzeyli tokenizer ve egitim/dogrulama bolmesi
├── train.py          # Egitim dongusu ve hiperparametreler
├── generate.py       # Komut satiri cikarimi
├── collect_lyrics.py # Genius API cekim scripti
├── app.py            # Gradio web arayuzu
├── requirements.txt
└── checkpoints/
    └── model.pt      # Kaydedilmis model agirliklari (git'e dahil edilmez)
```

---

## Notlar

- Model anlam bilmez; ogrendigi kaliplara dayanarak bir sonraki karakteri tahmin eder. Cikti kasten kusurlu ve deneyseldir.
- `lupe_dataset.txt` ve `checkpoints/model.pt` surum kontrolune dahil edilmez. Bunlari yerel olarak yeniden uretmek icin toplama ve egitim adimlarini calistir.
