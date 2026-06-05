import lyricsgenius
import re
import os

TOKEN = os.environ.get("GENIUS_TOKEN", "SENIN_TOKEN_BURAYA")

if TOKEN == "SENIN_TOKEN_BURAYA":
    print("HATA: GENIUS_TOKEN ortam degiskeni tanimlanmamis.")
    print("Kullanim: GENIUS_TOKEN=abc123 python collect_lyrics.py")
    exit(1)

genius = lyricsgenius.Genius(
    TOKEN,
    skip_non_songs=True,
    excluded_terms=["(Remix)", "(Live)", "(Instrumental)", "(Demo)"],
    remove_section_headers=True,
    timeout=15,
    retries=3,
)
genius.verbose = False

print("Lupe Fiasco sarkilari cekiliyor (bu birkaç dakika sürebilir)...")
artist = genius.search_artist("Lupe Fiasco", max_songs=150)

saved = 0
with open("lupe_dataset.txt", "w", encoding="utf-8") as f:
    for song in artist.songs:
        if not song.lyrics:
            continue
        lyrics = song.lyrics
        # Genius footer ve embed kalıntılarını temizle
        lyrics = re.sub(r'\d+EmbedYou might also like.*', '', lyrics, flags=re.DOTALL)
        lyrics = re.sub(r'See Lupe Fiasco Live.*', '', lyrics, flags=re.DOTALL)
        lyrics = re.sub(r'\[.*?\]', '', lyrics)          # [Verse 1] gibi başlıklar
        lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)       # Fazla boş satır
        lyrics = lyrics.strip()
        if len(lyrics) < 100:
            continue
        f.write(lyrics)
        f.write("\n\n---\n\n")
        saved += 1

print(f"Tamamlandi! {saved} sarki kaydedildi -> lupe_dataset.txt")
