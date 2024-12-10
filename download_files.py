import gdown
import os

def download_file(file_url, destination_path):
    """Google Driveからファイルをダウンロードする関数"""
    if not os.path.exists(destination_path):
        print(f"Downloading {destination_path}...")
        gdown.download(file_url, destination_path, quiet=False)
    else:
        print(f"{destination_path} already exists.")

def download_files():
    """必要なファイルをダウンロードする関数"""
    # Google DriveのファイルIDを指定（共有リンクからIDを抽出）
    sample_url = "https://drive.google.com/uc?id=1Egf77YksEqd2sDLdTyjC__VNgmvJFqdg"
    npy_url = "https://drive.google.com/uc?id=1rZl3xUMDt4iePMYd0Tek1df4Orwm73Af"

    # ダウンロード先のパスを指定
    sample_pt_path = "myproject/sample.pt"
    model_npy_path = "myproject/lyrics_fasttext_model.model.wv.vectors_ngrams.npy"
    
    # 各ファイルをダウンロード
    download_file(sample_url, sample_pt_path)
    download_file(npy_url, model_npy_path)

if __name__ == "__main__":
    download_files()
