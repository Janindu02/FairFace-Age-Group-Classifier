# FairFace Age Group Classifier — Streamlit Demo

**CAME Individual Assignment 1**

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web application |
| `fairface_cnn_recommended.pt` | Trained model weights (export from notebook) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push these files to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point to `app.py` in your repo.
4. Set Python version to **3.10**.
5. Upload `fairface_cnn_recommended.pt` to the repo root (or set a Streamlit secret path).

> **Important:** The `fairface_cnn_recommended.pt` weight file must be present at the same
> directory level as `app.py`. Generate it by running the Jupyter Notebook in full and
> copying from `checkpoints/fairface_cnn_recommended.pt`.

## Model details

- Architecture: 4-block CNN (Conv → BatchNorm → ReLU → MaxPool) + GAP + Dropout
- Input: 64×64 RGB face image
- Output: 9 age group classes (FairFace labels)
- Mitigation: Balanced Mini-Batch Sampling (WeightedRandomSampler)
- Training: PyTorch, from scratch — no pretrained models used

## Limitations

This is a research prototype for academic evaluation only. Not suitable for
production, surveillance, or consequential decision-making.
