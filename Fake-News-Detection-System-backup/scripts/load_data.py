import pandas as pd
import string
import re

def load_dataset():
    # Load the dataset
    df = pd.read_csv("data/WELFake_Dataset.csv")
    
    # Text cleaning
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(f"[{string.punctuation}]", "", text)
            return text
        return ""

    df['text'] = df['text'].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_dataset()
    print("Dataset loaded successfully!")
    print(f"Shape of dataset: {df.shape}")
    print("First 5 rows:")
    print(df.head())
