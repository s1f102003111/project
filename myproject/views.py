import pandas as pd
from django.shortcuts import render
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import lyricsgenius
from gensim.models import FastText
import re
import random
from pathlib import Path
from django.conf import settings

# FastTextモデルの読み込み
fasttext_model_path = Path(settings.BASE_DIR, "myproject", "lyrics_fasttext_model.model")
fasttext_model = FastText.load(str(fasttext_model_path))

# トークナイザーと感情分析モデルの読み込み
model_path = Path(settings.BASE_DIR, "myproject", "sample.pt")
checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
emotion_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=8)
emotion_model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

# Genius APIトークン
GENIUS_API_TOKEN = 'lqkYU2BX3cEkJjeQDO_kTXXi6CyWPtdzTpchWlU9Jpbze6v8OoL7JQOZ256evxOz'
genius = lyricsgenius.Genius(GENIUS_API_TOKEN)

# 感情のラベル
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']

# ソフトマックス関数
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

# 歌詞のクリーニング関数
def clean_lyrics(lyrics):
    patterns = [
        r'\d+\s+Contributors?.*?Lyrics.*?歌詞\]?',
        r'\bYou might also like\b',
        r'(?<=\S)\s*Embed\b',
        r'\d+Embed\b',
        r'\[.*?\]',
        r'\bYou might also likeEmbed\b',
    ]
    for pattern in patterns:
        lyrics = re.sub(pattern, '', lyrics)
    lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    return lyrics

# 感情分析関数
def analyze_lyrics_emotion(lyrics):
    emotion_model.eval()
    cleaned_lyrics = clean_lyrics(lyrics)
    tokens = tokenizer(cleaned_lyrics, truncation=True, return_tensors="pt")
    tokens.to(emotion_model.device)
    preds = emotion_model(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
    total = sum(prob)
    percentages = [(emotion, (score / total) * 100) for emotion, score in zip(emotion_names, prob)]

    # 確率が高い上位3つの感情とその割合を取得
    top_3_emotions = sorted(percentages, key=lambda x: x[1], reverse=True)[:3]
    return top_3_emotions

# 歌詞から特徴ベクトルを生成
def get_lyrics_vector(lyrics):
    words = clean_lyrics(lyrics).split()
    word_vectors = [fasttext_model.wv[word] for word in words if word in fasttext_model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # 平均ベクトル
    else:
        return np.zeros(fasttext_model.vector_size)

# 類似楽曲を推薦
def recommend_songs(input_vector, target_emotions, dataset_path=Path(settings.BASE_DIR, "myproject", "lyrics_dataset.csv"), num_recommendations=3):
    dataset_path = str(dataset_path)
    df = pd.read_csv(dataset_path)

    # 入力ベクトルと類似単語の計算
    similar_words = [word for word, _ in fasttext_model.wv.most_similar(input_vector, topn=10)]
    
    def contains_similar_words(lyrics, words):
        if pd.isna(lyrics):  # 歌詞がNaNの場合はスキップ
            return False
        return any(word in lyrics for word in words)

    # 歌詞に類似単語が含まれ、かつ指定感情ラベルを持つ曲をフィルタリング
    filtered_songs = df[
        df['Lyrics'].apply(lambda lyrics: contains_similar_words(lyrics, similar_words)) &
        df['Emotion Label'].isin(target_emotions)
    ]

    # ランダムに指定数を選択
    if len(filtered_songs) > 0:
        recommendations = filtered_songs.sample(n=min(num_recommendations, len(filtered_songs)))
    else:
        recommendations = pd.DataFrame()  # 空のデータフレーム

    return recommendations.rename(columns={
        'Song Title': 'song_title',
        'Artist': 'artist',
        'Emotion Label': 'emotion_label'
    }).to_dict('records')

# Djangoビュー
def index(request):
    if request.method == 'POST':
        album_title = request.POST['album_title']
        artist_name = request.POST['artist_name']

        # アルバム情報を取得
        album = genius.search_album(album_title, artist_name)
        if album:
            song_details = []

            for track in album.tracks:
                song_title = track.song.title
                lyrics = track.song.lyrics
                top_3_emotions = analyze_lyrics_emotion(lyrics)  # 上位3つの感情と割合を取得
                target_emotions = [emotion for emotion, _ in top_3_emotions]
                feature_vector = get_lyrics_vector(lyrics)
                recommended_songs = recommend_songs(feature_vector, target_emotions)

                cleaned_lyrics = clean_lyrics(lyrics)  # クリーニング後の歌詞

                song_details.append({
                    'song_title': song_title,
                    'top_3_emotions': top_3_emotions,  # 上位3つの感情と割合
                    'recommended_songs': recommended_songs,  # 類似楽曲
                    'lyrics': lyrics,
                    'cleaned_lyrics': cleaned_lyrics,  # クリーニング後の歌詞
                })

            context = {
                'album_title': album_title,
                'artist_name': artist_name,
                'song_details': song_details,
            }
            return render(request, 'myproject/index.html', context)

    return render(request, 'myproject/index.html')
