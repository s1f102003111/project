import gdown
import os
from pathlib import Path
from django.conf import settings

def download_file(file_url, destination_path):
    """Google Driveからファイルをダウンロードする関数"""
    destination_path = Path(destination_path)  # Pathオブジェクトに変換
    if not destination_path.exists():
        print(f"Downloading {destination_path}...")
        gdown.download(file_url, str(destination_path), quiet=False)
    else:
        print(f"{destination_path} already exists.")

def download_files():
    sample_url = "https://drive.google.com/uc?id=1Egf77YksEqd2sDLdTyjC__VNgmvJFqdg"
    npy_url = "https://drive.google.com/uc?id=1rZl3xUMDt4iePMYd0Tek1df4Orwm73Af"

    # Render 上のパスを直接指定
    sample_pt_path = Path("/opt/render/project/src/myproject/sample.pt")
    model_npy_path = Path("/opt/render/project/src/myproject/lyrics_fasttext_model.model.wv.vectors_ngrams.npy")
    
    download_file(sample_url, sample_pt_path)
    download_file(npy_url, model_npy_path)


if __name__ == "__main__":
    download_files()
