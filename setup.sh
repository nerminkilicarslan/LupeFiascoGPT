#!/bin/bash
# LupeFiascoGPT - Hizli Kurulum

echo ""
echo "=== LupeFiascoGPT Kurulumu ==="
echo ""

cd "$(dirname "$0")"

# Sanal ortam
python3 -m venv venv
source venv/bin/activate

# Kutuphaneler
pip install --upgrade pip -q
pip install torch numpy lyricsgenius gradio

echo ""
echo "=== Kurulum tamamlandi ==="
echo ""
echo "Adimlar:"
echo "  1. Genius API token al: genius.com/api-clients"
echo "  2. Sarkilari cek:   GENIUS_TOKEN=abc123 python collect_lyrics.py"
echo "  3. Modeli egit:     python train.py"
echo "  4. Uygulamayi ac:   python app.py   ->  http://localhost:7860"
echo ""
