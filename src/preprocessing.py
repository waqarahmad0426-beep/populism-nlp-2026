"""
Basic preprocessing pipeline for political texts
- remove URLs, mentions, extra whitespace
- language detection + translation to English
- very simple tokenization / lemmatization
"""

import re
import pandas as pd
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

DetectorFactory.seed = 42
translator = GoogleTranslator(source='auto', target='en')
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove URLs, mentions, RT, extra spaces
    text = re.sub(r'http\S+|www\S+|@\w+|RT|#[^\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def translate_if_non_english(text: str, max_len=4500) -> str:
    if len(text) < 20:
        return text
    try:
        lang = detect(text[:200])
        if lang != 'en':
            # truncate very long texts (API limit)
            text_to_translate = text[:max_len]
            translated = translator.translate(text_to_translate)
            return translated
        return text
    except:
        return text


def lemmatize_text(text: str) -> str:
    tokens = word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]
    return " ".join(lemmas)


def preprocess_dataframe(df: pd.DataFrame, text_col='text') -> pd.DataFrame:
    print("Cleaning text ...")
    df[text_col] = df[text_col].apply(clean_text)

    print("Translating non-English texts ... (can be slow)")
    df['translated'] = df[text_col].apply(translate_if_non_english)

    print("Lemmatizing ...")
    df['lemmatized'] = df['translated'].apply(lemmatize_text)

    df = df[df['lemmatized'].str.strip() != ''].copy()
    return df


if __name__ == "__main__":
    # Example usage (you will usually call from notebook)
    print("This is a library module. Run from notebook.")