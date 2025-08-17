import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

def download_youtube_trending():
    # Ensure your kaggle.json API token is placed in ~/.kaggle/kaggle.json
    os.makedirs("/home/iamsy/Project/datasets", exist_ok=True)

    print("Downloading YouTube Trending Dataset (2025)...")
    
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "sebastianbesinski/youtube-trending-videos-2025-updated-daily",
        file_path="",  # Gets the latest CSV or default file from dataset
    )
    
    print("Download complete. Saving CSV...")
    
    output_path = "/home/iamsy/Project/datasets/youtube_trending.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Saved to: {output_path}")
    print("First 5 rows:\n", df.head())
