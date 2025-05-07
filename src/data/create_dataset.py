import json
import pandas as pd
from datasets import Dataset

emotions = ["anger", "sadness", "joy", "fear", "surprise"]

with open('../data/raw.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

emotions_df = pd.DataFrame(raw_data, columns=["text", "emotion"])
emotions_df["emotion"] = emotions_df["emotion"].replace("sad", "sadness")

dataset = Dataset.from_pandas(emotions_df)
dataset = dataset.rename_column("emotion", "label")

dataset.push_to_hub("AliAfkhamii/hf_emotion_generation_texts")
