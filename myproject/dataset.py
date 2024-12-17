from django.shortcuts import render
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import lyricsgenius
import csv
import re
from pathlib import Path
from django.conf import settings

# モデルとトークナイザーの読み込み
model_path = Path(settings.BASE_DIR, "myproject", "sample.pt")
checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=8)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

GENIUS_API_TOKEN = 'GLfZTHTAxmnBJsU9hRYrN9KxKXHPcq0A_ycDm3eVy31p2uXRPcV4tRpscvxTqrvv'
genius = lyricsgenius.Genius(GENIUS_API_TOKEN)

# 感情のラベル
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']

# ソフトマックス関数
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def clean_lyrics(lyrics):
    # メタデータを除去する正規表現パターンを定義
    patterns = [
        r'\d+\s+Contributors?.*?Lyrics.*?歌詞\]?',  # 「数字 + Contributors + 任意のテキスト + Lyrics + 任意で "歌詞" または "歌詞]"」
        r'\d+\s+Contributors?.*?Lyrics',  # 数字 + Contributors〇〇〇 で始まり「Lyrics」で終わるパターンを削除
        r'\bYou might also like\b',  # "You might also like" のみ削除し、続く歌詞は残す
        r'(?<=\S)\s*Embed\b',  # Embed のみ削除し、Embed直前の単語は保持
        r'\d+Embed\b',  # 数字 + Embed のパターンを削除
        r'\[.*?\]',  # [Verse 1: Motoo Fujiwara] のような部分を削除
        r'\bYou might also likeEmbed\b',
    ]
    
    # 定義したパターンで歌詞から不要な部分を除去
    for pattern in patterns:
        lyrics = re.sub(pattern, '', lyrics)
    
    # 不要な空白を削除してきれいに整形
    lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    
    return lyrics



# 歌詞の感情分析
def analyze_lyrics_emotion(lyrics):
    model.eval()
    # 歌詞から不要なメタデータを除去
    cleaned_lyrics = clean_lyrics(lyrics)
    tokens = tokenizer(cleaned_lyrics, truncation=True, return_tensors="pt")
    tokens.to(model.device)
    preds = model(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
    out_dict = {n: p for n, p in zip(emotion_names, prob)}
    highest_emotion = max(out_dict, key=out_dict.get)  # 最も高い感情ラベルを取得
    return highest_emotion

# 歌詞、感情ラベル、曲名をCSVファイルに追記保存する関数
def save_lyrics_with_emotions(artist_name, filename=Path(settings.BASE_DIR, "myproject", "lyrics_dataset.csv")):
    artist = genius.search_artist(artist_name, max_songs=10)
    
    if not artist:
        print(f"アーティスト {artist_name} が見つかりませんでした。")
        return
    
    # CSVファイルに追記モードで書き込む
    with open(filename, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        
        # ファイルが空の場合、ヘッダー行を書き込む
        file.seek(0, 2)  # ファイルの末尾に移動
        if file.tell() == 0:
            writer.writerow(["Song Title", "Artist", "Lyrics", "Emotion Label"])  # ヘッダー行
        
        for song in artist.songs:
            lyrics = song.lyrics.replace("\n", " ")  # 歌詞の改行をスペースに変換
            emotion_label = analyze_lyrics_emotion(lyrics)  # 感情分析
            writer.writerow([song.title, artist_name, lyrics, emotion_label])  # 曲名、アーティスト名、歌詞、感情ラベルを書き込む
            print(f"Song: {song.title} by {artist_name} -> Emotion: {emotion_label}")
    
    print(f"楽曲データと感情ラベルが {filename} に追記されました。")

# メインの実行部分
if __name__ == "__main__":
    while True:
        artist_name = input("アーティスト名を入力してください（終了するには 'exit' と入力）: ")
        if artist_name.lower() == 'exit':
            break
        save_lyrics_with_emotions(artist_name)