# Product Attribute Extraction

Extracts structured fashion attributes (category, silhouette, fabric,
neckline, sleeve, length, embellishment, color) from free-text product
descriptions using TF-IDF + LinearSVC, one classifier per attribute.

## 🎥 Demo Video

Download and watch the demo here:

[▶️ Product Attribute Extraction Demo](demo/demo.mp4)


## Folder Structure

```
product-attribute-extraction/
│
├── dataset/
│   └── product_attribute_dataset.csv
│
├── src/
│   ├── preprocess.py    # text cleaning, TF-IDF, train/test split
│   ├── train.py         # trains + saves all models and encoders
│
├── models/              # populated by train.py
│   ├── tfidf.pkl
│   ├── <attribute>.pkl
│   ├── <attribute>_encoder.pkl
│   └── evaluation.csv
│
├── api/
│   └── main.py          # FastAPI service (/extract endpoint)
│
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train

```bash
cd src
python train.py
```

This will:
- clean the `description` column of `dataset/product_attribute_dataset_200.csv`
- fit a shared TF-IDF vectorizer (unigrams + bigrams, 5000 features)
- train one `LinearSVC` model per attribute
- save every model + label encoder + the vectorizer to `../models/`
- write `../models/evaluation.csv` with accuracy / precision / recall / F1
  for each attribute on the held-out 20% test split


## 2. Serve predictions via API

```bash
cd api
uvicorn main:app --reload
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"description": "Elegant chiffon bridesmaid dress with V neckline, pleated bodice and sage color."}'
```

Response:

```json
{
  "category": "Bridesmaid Dress",
  "silhouette": "A-Line",
  "fabric": "Chiffon",
  "neckline": "V-Neck",
  "sleeve": "Sleeveless",
  "length": "Floor",
  "embellishment": "Pleats",
  "color": "Sage"
}
```

