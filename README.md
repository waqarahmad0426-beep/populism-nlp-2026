# Populism NLP 2026

## Quick Start – How to Run the Project

### 1. Clone the repository (or download ZIP)

```bash
git clone https://github.com/[your-username]/populism-nlp-2026.git
cd populism-nlp-2026
```

### 2. Create virtual environment & install packages

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download required NLTK data (one-time)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 4. Get the data (do this once)

Automatic download (Miller Center speeches):

```bash
python src/data_collection.py
```
