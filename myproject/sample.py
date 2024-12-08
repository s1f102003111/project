import pandas as pd
from gensim.models import FastText

# モデルをロード
model_path = "C:/Users/iniad/Documents/albumapp/myproject/lyrics_fasttext_model.model"
model = FastText.load(model_path)

# データセットをロード
csv_path = "C:/Users/iniad/Documents/albumapp/myproject/lyrics_dataset.csv"
df = pd.read_csv(csv_path)

# 指定された単語と類似単語を取得
target_word = "空"
similar_words = [word for word, _ in model.wv.most_similar(target_word, topn=10)]

# 感情ラベルの条件
target_emotions = {"Joy", "Disgust", "Sadness"}

# 歌詞に類似単語が含まれる曲をフィルタリング
def contains_similar_words(lyrics, words):
    if pd.isna(lyrics):  # 歌詞がNaNの場合はスキップ
        return False
    return any(word in lyrics for word in words)

filtered_songs = df[
    df['Lyrics'].apply(lambda lyrics: contains_similar_words(lyrics, similar_words)) &
    df['Emotion Label'].isin(target_emotions)
]

# 結果を表示
if not filtered_songs.empty:
    print("以下の曲が指定された単語または類似単語を含み、指定の感情ラベルを持っています：")
    for index, row in filtered_songs.iterrows():
        print(f"曲名: {row['Song Title']}, アーティスト名: {row['Artist']}, 感情ラベル: {row['Emotion Label']}")
else:
    print("指定された条件を満たす曲は見つかりませんでした。")
