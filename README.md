# Sentiment Analysis for Google Classroom Reviews

A lightweight, notebook-driven pipeline to scrape Google Play Store reviews for Google Classroom, preprocess English text, and run sentiment inference using two models: Logistic Regression (with TF‑IDF features) and a Dense Neural Network (DNN). The repo includes ready-to-run notebooks for scraping and inference, plus pretrained artifacts in the project root.

## Features

- Scrape Google Play reviews and export to CSV (`ulasan_classroom.csv`).
- Robust text preprocessing: cleaning, lowercasing, slang normalization, tokenization, stopword removal, lemmatization.
- Dual-model predictions: Logistic Regression (TF‑IDF) and Dense Neural Network.
- Interactive, CLI-style input inside the inference notebook.
- Clear, reproducible environment via `requirements.txt`.

## Tech Stack

- Python
- Data & NLP: NumPy, pandas, NLTK, scikit‑learn, joblib
- Deep Learning: TensorFlow/Keras
- Scraping: google‑play‑scraper
- Visualization (optional): matplotlib, seaborn, wordcloud
- Notebooks: Jupyter / VS Code Notebook

## Key Files

- Notebooks:
  - Inference: `Inference_Nelson_Ahli.ipynb`
  - Scraping: `Scraping_Nelson_Ahli.ipynb`
- Models & vectorizer (pretrained):
  - `TFIDF_model_sentiment.pkl`
  - `LR_model_sentiment.pkl`
  - `DNN_model_sentiment.h5`
- Data sample:
  - `ulasan_classroom.csv`
- Project manifest:
  - `requirements.txt`

## Installation

1) Create and activate a virtual environment (recommended)

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

The pinned dependencies include (from `requirements.txt`):

- `google_play_scraper`, `joblib`, `keras`, `matplotlib`, `nltk`, `numpy`, `pandas`, `protobuf`, `requests`, `scikit_learn`, `seaborn`, `tensorflow`, `wordcloud`

3) Download NLTK resources (first run)

The inference notebook will attempt to download NLTK corpora at runtime. To pre-download:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

4) Ensure model artifacts are available

This repository already includes the pretrained artifacts in the project root:

- `TFIDF_model_sentiment.pkl`
- `LR_model_sentiment.pkl`
- `DNN_model_sentiment.h5`

If you move them, adjust paths in the inference notebook accordingly.

5) Adjust Colab-style paths (if needed)

Inside `Inference_Nelson_Ahli.ipynb`, the model/vectorizer are loaded from `/content/...` (a Google Colab path). For local runs, change these to relative paths, for example:

```python
tfidf = joblib.load("TFIDF_model_sentiment.pkl")
model_lr = joblib.load("LR_model_sentiment.pkl")
model_dnn = tf.keras.models.load_model("DNN_model_sentiment.h5")
```

## How to Run

Option A — VS Code Notebook
- Open `Inference_Nelson_Ahli.ipynb` in VS Code.
- Select the Python interpreter for your virtual environment.
- Run all cells (or step through them).
- At the input prompt, type a review (e.g., "this app is amazing") and view predictions from both models.

Option B — Jupyter (CLI)

```bash
python -m jupyter notebook
# or
python -m jupyter lab
```

- Open `Inference_Nelson_Ahli.ipynb` and run all cells.

Option C — Scrape dataset
- Open and run `Scraping_Nelson_Ahli.ipynb`.
- It uses `google_play_scraper` to fetch reviews for Google Classroom (`com.google.android.apps.classroom`) and saves to `ulasan_classroom.csv`.

## Workflow Overview

1) Scraping
- Use `Scraping_Nelson_Ahli.ipynb` to fetch and save reviews into `ulasan_classroom.csv`.

2) Preprocessing & Inference
- The inference notebook:
	- Cleans and normalizes text
	- Replaces English slang using a CSV dictionary fetched from GitHub
	- Tokenizes, removes stopwords, and lemmatizes
	- Transforms with TF‑IDF
	- Predicts sentiment using Logistic Regression and DNN

3) Outputs
- Console output shows the predicted sentiment for each model.

## Notes & Troubleshooting

- NLTK lookups: If you see lookup errors for `stopwords`, `wordnet`, or `punkt`, run the download commands in the Installation section.
- TensorFlow GPU: Not required; the DNN will run on CPU by default.
- Internet access: The slang dictionary is fetched over HTTP during preprocessing. If offline, cache it locally or replace with a local CSV.
- Path issues: If you keep artifacts in subfolders, ensure the notebook load paths point to the correct locations.

---

Maintained for the BFDL module — notebooks and models by Nelson Ahli.
