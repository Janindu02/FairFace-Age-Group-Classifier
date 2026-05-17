"""
FairFace Age Group Classifier — Streamlit Demo
CAME Individual Assignment 1

Architecture: 5-block CNN with residual skip connection
(matches fairface_cnn_recommended.pt exactly)
"""

import os
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
AGE_NAMES = [
    "0-2", "3-9", "10-19", "20-29", "30-39",
    "40-49", "50-59", "60-69", "70+"
]
AGE_EMOJIS = ["👶", "🧒", "🧑", "👨", "🧔", "🧑‍🦳", "👴", "🧓", "🧓"]
AGE_COLORS = [
    "#FF6B6B", "#FF9F43", "#FECA57", "#48DBFB",
    "#1DD1A1", "#54A0FF", "#5F27CD", "#C8D6E5", "#8395A7"
]

NUM_CLASSES = len(AGE_NAMES)
INPUT_SIZE  = 64
MEAN        = (0.485, 0.456, 0.406)
STD         = (0.229, 0.224, 0.225)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fairface_cnn_recommended.pt")

# ─────────────────────────────────────────────────────────────────────────────
# Model — matches saved checkpoint exactly
# Keys in .pt: block1-5, skip4, classifier.2, classifier.5
# ─────────────────────────────────────────────────────────────────────────────
class FairFaceCNN(nn.Module):
    """
    5-block CNN:
      Blocks 1-4 : Conv(k=3,p=1) → BN → ReLU → MaxPool(2)
      Block 5    : Conv(k=3,p=1) → BN → ReLU          (no pool)
      skip4      : Conv(1×1) projection 256→512
      forward    : x5 = block5(x4) + skip4(x4)
      Head       : GAP → Flatten → Dropout(0.5) → Linear(512,128)
                   → ReLU → Dropout(0.25) → Linear(128,9)
    """
    def __init__(self, num_classes=9, dropout_p=0.5):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.block1 = conv_block(3,   32)
        self.block2 = conv_block(32,  64)
        self.block3 = conv_block(64,  128)
        self.block4 = conv_block(128, 256)

        # Block 5 — no MaxPool (kept same spatial size as block4 output)
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # 1×1 conv to match channels for residual add
        self.skip4 = nn.Conv2d(256, 512, kernel_size=1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x  = self.block1(x)
        x  = self.block2(x)
        x  = self.block3(x)
        x4 = self.block4(x)
        x  = self.block5(x4) + self.skip4(x4)
        x  = self.gap(x)
        x  = self.classifier(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached across sessions)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FairFaceCNN(num_classes=NUM_CLASSES).to(device)
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model weights not found at `{MODEL_PATH}`.")
        st.stop()
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    # Handle both raw state_dict and wrapped checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def predict_all(pil_image):
    """Return list of (age_name, probability_pct) for all 9 classes, in age order."""
    model, device = load_model()
    tensor = _transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].cpu().tolist()
    return [(AGE_NAMES[i], round(probs[i] * 100, 2)) for i in range(NUM_CLASSES)]


# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark sci-fi / editorial aesthetic
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg:       #090D18;
    --s1:       #111827;
    --s2:       #1A2438;
    --border:   #243050;
    --accent:   #4F8EF7;
    --green:    #1DD1A1;
    --text:     #E8EDF5;
    --muted:    #7A8BA0;
    --r:        14px;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main { background: var(--bg) !important; }
[data-testid="stHeader"]  { background: transparent !important; display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

* { font-family: 'DM Sans', sans-serif !important; color: var(--text); box-sizing: border-box; }

.block-container { padding: 1.5rem 1.5rem 4rem !important; max-width: 860px !important; }

/* ── Hero ── */
.hero { text-align: center; padding: 2.5rem 1rem 1.8rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(79,142,247,.1); border: 1px solid rgba(79,142,247,.25);
    border-radius: 999px; padding: .3rem .9rem;
    font-size: .68rem; font-weight: 700; letter-spacing: .22em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 1.2rem;
}
.hero-badge::before { content: '●'; color: var(--accent); }
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2rem,5vw,3rem) !important;
    font-weight: 800 !important; line-height: 1.1 !important;
    background: linear-gradient(140deg,#E8EDF5 35%,#4F8EF7 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 .7rem !important;
}
.hero-sub { color: var(--muted); font-size: .97rem; line-height: 1.65; max-width: 500px; margin: 0 auto; }

/* ── Upload area ── */
[data-testid="stFileUploader"] > div {
    background: var(--s1) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--r) !important;
    transition: border-color .2s, background .2s !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--accent) !important;
    background: rgba(79,142,247,.04) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] svg { fill: var(--accent) !important; }
.st-emotion-cache-7ym5gk, .uploadedFileName { color: var(--accent) !important; }

/* ── Card ── */
.card {
    background: var(--s1); border: 1px solid var(--border);
    border-radius: var(--r); padding: 1.4rem; margin-bottom: .9rem;
}
.card-label {
    font-family: 'Syne', sans-serif !important;
    font-size: .65rem; font-weight: 700; letter-spacing: .2em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 1rem;
}

/* ── Top result ── */
.winner {
    background: linear-gradient(135deg, var(--s2) 0%, rgba(79,142,247,.07) 100%);
    border: 1px solid rgba(79,142,247,.28); border-radius: var(--r);
    padding: 2rem 1.5rem; text-align: center; margin-bottom: 1rem;
    position: relative; overflow: hidden;
}
.winner::after {
    content:''; position:absolute; top:-80px; right:-80px;
    width:220px; height:220px;
    background: radial-gradient(circle,rgba(79,142,247,.12),transparent 70%);
    pointer-events:none;
}
.winner-emoji { font-size:3rem; display:block; margin-bottom:.5rem; }
.winner-age {
    font-family:'Syne',sans-serif !important;
    font-size:2.2rem; font-weight:800; color:var(--text); margin-bottom:.2rem;
}
.winner-conf { font-size:.95rem; color:var(--accent); font-weight:500; }

/* ── Top-3 row ── */
.t3 { display:flex; gap:.7rem; margin-bottom:.9rem; }
.t3-card {
    flex:1; background:var(--s2); border-radius:12px;
    padding:1rem .7rem; text-align:center;
}
.t3-emoji { font-size:1.6rem; margin-bottom:.3rem; }
.t3-age { font-family:'Syne',sans-serif !important; font-size:.92rem; font-weight:800; margin-bottom:.15rem; }
.t3-pct { font-size:.78rem; font-weight:600; }
.t3-rank { font-size:.62rem; color:var(--muted); margin-top:.15rem; }

/* ── Bars ── */
.bars { display:flex; flex-direction:column; gap:.55rem; }
.b-row { display:flex; align-items:center; gap:.7rem; }
.b-lbl { width:52px; font-size:.78rem; color:var(--muted); text-align:right; flex-shrink:0; }
.b-track { flex:1; height:7px; background:var(--s2); border-radius:999px; overflow:hidden; }
.b-fill  { height:100%; border-radius:999px; }
.b-val   { width:42px; font-size:.78rem; font-weight:500; flex-shrink:0; }
.b-row.top .b-lbl { color:var(--text); font-weight:600; }
.b-row.top .b-val { color:var(--accent); font-weight:700; }

/* ── Divider ── */
hr { border:none !important; border-top:1px solid var(--border) !important; margin:1.8rem 0 !important; }

/* ── Expander override ── */
[data-testid="stExpander"] {
    background:var(--s1) !important; border:1px solid var(--border) !important;
    border-radius:var(--r) !important;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] li { color:var(--muted) !important; font-size:.88rem !important; line-height:1.65 !important; }
[data-testid="stExpander"] strong { color:var(--text) !important; }
[data-testid="stExpander"] a { color:var(--accent) !important; }

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius:var(--r) !important; border:1px solid var(--border) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ── Footer ── */
.ft { text-align:center; color:var(--muted); font-size:.75rem; line-height:2; padding:2rem 0 .5rem; }
.ft strong { color:var(--text); }
.ft a { color:var(--accent) !important; text-decoration:none; }

/* ── Alert ── */
[data-testid="stAlert"] {
    background:var(--s1) !important; border:1px solid var(--border) !important;
    border-radius:var(--r) !important;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# HTML component helpers
# ─────────────────────────────────────────────────────────────────────────────
def html_winner(age, conf, emoji):
    return (f'<div class="winner">'
            f'<span class="winner-emoji">{emoji}</span>'
            f'<div class="winner-age">Age {age}</div>'
            f'<div class="winner-conf">{conf:.1f}% confidence</div>'
            f'</div>')


def html_top3(results):
    top3 = sorted(results, key=lambda x: x[1], reverse=True)[:3]
    inner = ""
    for rank, (age, conf) in enumerate(top3):
        i     = AGE_NAMES.index(age)
        color = AGE_COLORS[i]
        inner += (f'<div class="t3-card" style="border-top:3px solid {color};opacity:{1-rank*0.2:.1f};">'
                  f'<div class="t3-emoji">{AGE_EMOJIS[i]}</div>'
                  f'<div class="t3-age" style="color:var(--text)">{age}</div>'
                  f'<div class="t3-pct" style="color:{color}">{conf:.1f}%</div>'
                  f'<div class="t3-rank">#{rank+1}</div>'
                  f'</div>')
    return f'<div class="card"><div class="card-label">Top 3 Predictions</div><div class="t3">{inner}</div></div>'


def html_bars(results):
    top_conf = max(c for _, c in results)
    rows = ""
    for i, (age, conf) in enumerate(results):
        is_top  = abs(conf - top_conf) < 0.01
        pct     = conf / top_conf * 100 if top_conf > 0 else 0
        color   = AGE_COLORS[i]
        cls     = "b-row top" if is_top else "b-row"
        rows += (f'<div class="{cls}">'
                 f'<span class="b-lbl">{age}</span>'
                 f'<div class="b-track"><div class="b-fill" style="width:{pct:.1f}%;background:{color};"></div></div>'
                 f'<span class="b-val">{conf:.1f}%</span>'
                 f'</div>')
    return (f'<div class="card">'
            f'<div class="card-label">Full Probability Distribution</div>'
            f'<div class="bars">{rows}</div>'
            f'</div>')


# ─────────────────────────────────────────────────────────────────────────────
# App entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="FairFace Age Classifier",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CSS, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">CAME Assignment 1 &nbsp;·&nbsp; FairFace CNN</div>
        <h1>Age Group Classifier</h1>
        <div class="hero-sub">
            Upload any face photo — JPG, PNG, WEBP, BMP, GIF or TIFF — and a CNN
            trained from scratch on FairFace will predict the age group with a
            full probability breakdown across all 9 classes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        label="upload",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif"],
        help="Supports JPG, PNG, WEBP, BMP, TIFF, GIF — any resolution",
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if uploaded is not None:

        # Open image safely — handle animated GIFs, unusual modes, etc.
        try:
            raw = Image.open(uploaded)
            if getattr(raw, "n_frames", 1) > 1:
                raw.seek(0)
            pil_img = raw.convert("RGB")
        except Exception as err:
            st.error(f"Could not read image file: {err}. Please try a different file.")
            st.stop()

        # Run inference
        with st.spinner("Analysing image…"):
            try:
                results = predict_all(pil_img)
            except Exception as err:
                st.error(f"Prediction failed: {err}")
                st.stop()

        top_age, top_conf = max(results, key=lambda x: x[1])
        top_emoji = AGE_EMOJIS[AGE_NAMES.index(top_age)]

        # ── Two-column layout ─────────────────────────────────────────────────
        col_l, col_r = st.columns([1, 1], gap="large")

        with col_l:
            st.image(pil_img, use_column_width=True, caption="Uploaded image")

        with col_r:
            st.markdown(html_winner(top_age, top_conf, top_emoji), unsafe_allow_html=True)
            st.markdown(html_top3(results), unsafe_allow_html=True)

        # ── Full bar chart below ───────────────────────────────────────────────
        st.markdown(html_bars(results), unsafe_allow_html=True)

        # ── Raw numbers expander ──────────────────────────────────────────────
        with st.expander("📊 All class probabilities"):
            ca, cb = st.columns(2)
            half = NUM_CLASSES // 2 + 1
            with ca:
                for age, conf in results[:half]:
                    st.markdown(
                        f"<span style='color:var(--muted);font-size:.88rem;'>{age}</span>"
                        f"&ensp;<strong style='color:var(--text)'>{conf:.2f}%</strong>",
                        unsafe_allow_html=True,
                    )
            with cb:
                for age, conf in results[half:]:
                    st.markdown(
                        f"<span style='color:var(--muted);font-size:.88rem;'>{age}</span>"
                        f"&ensp;<strong style='color:var(--text)'>{conf:.2f}%</strong>",
                        unsafe_allow_html=True,
                    )

    else:
        # Empty state placeholder
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem 2rem;">
            <div style="font-size:3.5rem;margin-bottom:1rem;filter:grayscale(.4);">📷</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
                        margin-bottom:.5rem;">No image uploaded yet</div>
            <div style="color:var(--muted);font-size:.88rem;max-width:360px;
                        margin:0 auto;line-height:1.7;">
                Use the upload box above — drag and drop or click to browse.<br>
                Any face photo works: JPG, PNG, WEBP, BMP, GIF, TIFF.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── About expander ────────────────────────────────────────────────────────
    with st.expander("ℹ️ About this model"):
        st.markdown("""
**What it does**  
Predicts which of 9 age groups a person belongs to from a single face image:
`0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`.

**Architecture**  
5-block CNN (Conv → BatchNorm → ReLU → MaxPool) with a residual skip connection
on block 4→5, Global Average Pooling, and a 2-layer head (512→128→9).
Trained **from scratch** in PyTorch — no pretrained weights used.

**Fairness**  
Audited across 7 race groups and 2 gender groups using the
[FairFace dataset](https://huggingface.co/datasets/HuggingFaceM4/FairFace).
Trained with **Balanced Mini-Batch Sampling** (WeightedRandomSampler)
to reduce the race accuracy gap from 8.14 pp → 6.28 pp.

**Performance**  
Overall test accuracy: ~41.85% on 10,954 official FairFace validation samples.

**⚠️ Limitations & Responsible Use**  
- Research prototype — not a production system.  
- Performance is lower on age extremes (infants, elderly).  
- Must **not** be used for surveillance, identity verification, or consequential
  decisions without proper ethical and legal review (PDPA Sri Lanka / GDPR).
""")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="ft">
        <strong>FairFace CNN Age Group Classifier</strong><br>
        CAME Individual Assignment 1 &nbsp;·&nbsp;
        <a href="https://huggingface.co/datasets/HuggingFaceM4/FairFace" target="_blank">
            FairFace dataset</a> (config 0.25) &nbsp;·&nbsp;
        PyTorch CNN from scratch &nbsp;·&nbsp; Bias mitigation: Balanced Mini-Batch Sampling<br>
        <span style="color:var(--accent)">Institute of Java and Software Engineering (IJSE)</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
