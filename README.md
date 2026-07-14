# Product Attribute Extraction

Extracts structured fashion attributes (category, silhouette, fabric,
neckline, sleeve, length, embellishment, color) from free-text product
descriptions using TF-IDF + LinearSVC, one classifier per attribute.

## Folder Structure

```
product-attribute-extraction/
│
├── dataset/
│   └── product_attribute_dataset_200.csv   # 200-row sample dataset
│
├── src/
│   ├── preprocess.py    # text cleaning, TF-IDF, train/test split
│   ├── train.py         # trains + saves all models and encoders
│   └── predict.py       # CLI: predict attributes for one description
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

## 2. Predict from the command line

```bash
cd src
python predict.py "Elegant chiffon bridesmaid dress with V neckline, pleated bodice and sage color."
```

Example output:

```
Extracted Attributes:
  category       : Bridesmaid Dress
  silhouette     : A-Line
  fabric         : Chiffon
  neckline       : V-Neck
  sleeve         : Sleeveless
  length         : Floor
  embellishment  : Pleats
  color          : Sage
```

## 3. Serve predictions via API

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

## Notes on the sample dataset

`dataset/product_attribute_dataset_200.csv` here is a **synthetically
generated** 200-row dataset (templated descriptions) so the whole
pipeline runs end-to-end out of the box. Swap in your real labeled
dataset (same column names: `description`, `category`, `silhouette`,
`fabric`, `neckline`, `sleeve`, `length`, `embellishment`, `color`) and
re-run `train.py` — the near-perfect scores you'd see on the synthetic
data are an artifact of its templated, low-noise nature and are not
representative of accuracy on real-world listings.
