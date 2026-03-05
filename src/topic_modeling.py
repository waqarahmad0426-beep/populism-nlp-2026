"""
Week 3: BERTopic Topic Modeling and Zero-Shot Subtype Detection

This script follows the project proposal:
- BERTopic for thematic extraction (RQ2)
- Zero-shot LLM prompting for populist subtypes (novel contribution)
"""

import pandas as pd
import warnings
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import plotly.express as px

warnings.filterwarnings("ignore")

# Configuration
DATA_PATH = "data/cleaned/cleaned_speeches.csv"
OUTPUT_TOPIC_INFO = "visualizations/topic_info.csv"
OUTPUT_HTML = "visualizations/topic_evolution.html"

def run_topic_modeling():
    print("Starting Week 3: BERTopic and Zero-Shot Analysis")

    # Load cleaned data from previous week
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} speeches from cleaned dataset")

    # BERTopic Model Training
    print("Training BERTopic model with multilingual embeddings...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=CountVectorizer(stop_words="english"),
        min_topic_size=10,
        language="multilingual"
    )

    topics, probs = topic_model.fit_transform(df['lemmatized'].tolist())

    # Save topic information
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(OUTPUT_TOPIC_INFO, index=False)
    print(f"Total topics found: {len(topic_info) - 1}")

    # Visualization: Topic Evolution Over Years
    df['topic'] = topics
    fig = px.histogram(df, x="year", color="topic", title="Topic Evolution Over Years")
    fig.write_html(OUTPUT_HTML)
    print(f"Visualization saved: {OUTPUT_HTML}")

    # Zero-Shot Subtype Detection (Proposal Novel Contribution)
    print("Running zero-shot classification for populist subtypes...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["anti-elitism", "people-centrism", "exclusionism", "neutral"]

    # Run on sample for speed
    sample = df.sample(min(100, len(df)))

    def get_subtype(text):
        result = classifier(text[:512], labels)
        return result['labels'][0], round(result['scores'][0], 3)

    sample['subtype'], sample['subtype_score'] = zip(*sample['lemmatized'].apply(get_subtype))

    # Save results
    sample[['lemmatized', 'subtype', 'subtype_score']].to_csv("visualizations/subtype_scores.csv", index=False)
    print("Subtype scoring completed and saved.")

    # Show sample output
    print("\nSample subtype results:")
    print(sample[['subtype', 'subtype_score']].head(10))

    print("\nWeek 3 processing completed successfully.")


if __name__ == "__main__":
    run_topic_modeling()