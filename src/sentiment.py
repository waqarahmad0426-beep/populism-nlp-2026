"""
Very first sentiment analysis helpers
(VADER + TextBlob – baselines before more advanced models)
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd


analyzer = SentimentIntensityAnalyzer()


def add_vader_scores(df: pd.DataFrame, text_col='lemmatized') -> pd.DataFrame:
    def get_scores(text):
        s = analyzer.polarity_scores(text)
        return pd.Series({
            'vader_compound': s['compound'],
            'vader_pos': s['pos'],
            'vader_neg': s['neg'],
            'vader_neu': s['neu']
        })

    scores = df[text_col].apply(get_scores)
    return pd.concat([df, scores], axis=1)


def add_textblob_scores(df: pd.DataFrame, text_col='lemmatized') -> pd.DataFrame:
    def get_tb_score(text):
        blob = TextBlob(text)
        return pd.Series({
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        })

    scores = df[text_col].apply(get_tb_score)
    return pd.concat([df, scores], axis=1)


if __name__ == "__main__":
    print("Sentiment helpers loaded. Use add_vader_scores() and add_textblob_scores()")