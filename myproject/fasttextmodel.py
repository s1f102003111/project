import pandas as pd
import re
from gensim.models import FastText
import MeCab
from stopwordsiso import stopwords

# 歌詞データセットの読み込み
dataset_path = 'C:/Users/iniad/Documents/albumapp/myproject/lyrics_dataset.csv'
df = pd.read_csv(dataset_path)

# MeCabとNEologdの設定
mecab = MeCab.Tagger(r'-d "C:\\Program Files\\MeCab\\dic\\ipadic" -u "C:\\Program Files\\MeCab\\dic\\NEologd\\NEologd.20200910-u.dic"')

# 定義済みの日本語ストップワードを取得
stop_words = stopwords("ja")

# 不要なデータの除去とストップワード除去、トークン化
def clean_lyrics(lyrics):
    # メタデータの削除
    patterns = [
        r'\d+\s+Contributors?.*?Lyrics.*?歌詞\]?',  
        r'\bYou might also like\b',  
        r'(?<=\S)\s*Embed\b',  
        r'\d+Embed\b',  
        r'\[.*?\]',  
        r'\bYou might also likeEmbed\b',
        r'[a-zA-Z]+',  # 英語部分を削除するパターン
    ]
    for pattern in patterns:
        lyrics = re.sub(pattern, '', lyrics)
    
    # MeCabで形態素解析とストップワード除去
    tokens = []
    node = mecab.parseToNode(lyrics)
    while node:
        word = node.surface
        pos = node.feature.split(',')[0]  # 品詞情報（名詞、動詞、形容詞など）
        if word and word not in stop_words and pos in {'名詞', '動詞', '形容詞'}:
            tokens.append(word)
        node = node.next

    return tokens

# 歌詞のクリーニングとトークン化
cleaned_lyrics = [clean_lyrics(lyrics) for lyrics in df['Lyrics']]

# FastTextモデルの学習
model = FastText(sg=1, vector_size=100, window=5, min_count=5)
model.build_vocab(corpus_iterable=cleaned_lyrics)
model.train(corpus_iterable=cleaned_lyrics, total_examples=len(cleaned_lyrics), epochs=10)

# 学習したモデルの保存
model.save('lyrics_fasttext_model.model')
