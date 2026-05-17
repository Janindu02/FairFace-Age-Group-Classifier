"""
FairFace Age Group Classifier — Streamlit Demo
CAME Individual Assignment 1

ARCHITECTURE: 5-block CNN + residual skip (matches checkpoint keys exactly)
  block1-4 : Conv(k=3,p=1,bias=False) → BN → ReLU → MaxPool(2)
  block5   : Conv(k=3,p=1,bias=False) → BN → ReLU  (no pool)
  skip4    : Conv(1×1, bias=False)  [256→512 projection]
  forward  : out = block5(x4) + skip4(x4)
  head     : GAP → Flatten → Dropout(0.5) → Linear(512,128)
             → ReLU → Dropout(0.25) → Linear(128,9)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageStat
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
AGE_NAMES  = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"]
AGE_EMOJI  = ["👶","🧒","🧑","👨","🧔","🧑‍🦳","👴","🧓","🧓"]
AGE_COLOR  = ["#FF6B6B","#FF9F43","#FECA57","#48DBFB",
              "#1DD1A1","#54A0FF","#7C3AED","#C8D6E5","#8395A7"]
NUM_CLS    = 9
INPUT_SIZE = 64
MEAN, STD  = (0.485,0.456,0.406), (0.229,0.224,0.225)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fairface_cnn_recommended.pt")

# ── Model (exact match to checkpoint) ─────────────────────────────────────────
class FairFaceCNN(nn.Module):
    def __init__(self, num_classes=9, dropout_p=0.5):
        super().__init__()
        def blk(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2))
        self.block1 = blk(3,   32)
        self.block2 = blk(32,  64)
        self.block3 = blk(64,  128)
        self.block4 = blk(128, 256)
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.skip4  = nn.Conv2d(256, 512, 1, bias=False)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x4 = self.block4(x)
        x = self.block5(x4) + self.skip4(x4)
        return self.classifier(self.gap(x))


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = FairFaceCNN(num_classes=NUM_CLS).to(device)
    if not os.path.exists(MODEL_PATH):
        return None, device, "weights_missing"
    try:
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        m.load_state_dict(state)
        m.eval()
        # Diagnose whether weights are trained or near-init
        w = state.get("classifier.5.weight",
                      state.get("classifier.5.weight", None))
        if w is None:
            return m, device, "loaded"
        w_std = w.std().item()
        # Xavier init for (128,9) gives std ≈ 0.118;  trained model >> 0.25
        status = "undertrained" if w_std < 0.20 else "loaded"
        return m, device, status
    except Exception as e:
        return None, device, f"error:{e}"


# ── Inference ──────────────────────────────────────────────────────────────────
_tf = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

def predict(pil_img):
    """Returns list of (age_name, pct) in age order."""
    model, device, status = load_model()
    if model is None:
        return None, status
    tensor = _tf(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].cpu().tolist()
    return [(AGE_NAMES[i], round(probs[i]*100, 2)) for i in range(NUM_CLS)], status


# ── Simple face/content heuristic ─────────────────────────────────────────────
def looks_like_face(pil_img):
    """
    Very lightweight heuristic: skin-tone pixel ratio check.
    Returns (bool, confidence_str).
    Not a real face detector — just filters obviously non-face images.
    """
    rgb = pil_img.convert("RGB").resize((64, 64))
    arr = np.array(rgb, dtype=float)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    # Skin tone: R > 60, G > 40, B > 20, R > G > B, R-B > 15
    skin = (r > 60) & (g > 40) & (b > 20) & (r > g) & (g > b) & ((r - b) > 15)
    ratio = skin.sum() / (64 * 64)
    return ratio > 0.10, ratio


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:      #07101F;
    --s1:      #0F1A2E;
    --s2:      #162035;
    --border:  #1F3050;
    --accent:  #4A90F5;
    --green:   #10B981;
    --amber:   #F59E0B;
    --red:     #EF4444;
    --text:    #E2EAF4;
    --muted:   #6B8099;
    --r:       12px;
    --font:    'DM Sans', system-ui, sans-serif;
    --head:    'Syne', system-ui, sans-serif;
}

html, body { background: var(--bg) !important; }
[data-testid="stAppViewContainer"]      { background: var(--bg) !important; }
[data-testid="stAppViewContainer"] .main{ background: var(--bg) !important; }
[data-testid="stHeader"]                { display: none !important; }
[data-testid="stDecoration"]            { display: none !important; }
[data-testid="stToolbar"]               { display: none !important; }
#MainMenu, footer                       { display: none !important; }

.block-container {
    font-family: var(--font) !important;
    color: var(--text) !important;
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 820px !important;
}

/* ── All text in app ── */
p, span, div, li, label { color: var(--text) !important; font-family: var(--font) !important; }

/* ── Hero ── */
.hero {
    padding: 3rem 0 2rem;
    text-align: center;
}
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(74,144,245,.12);
    border: 1px solid rgba(74,144,245,.3);
    border-radius: 999px; padding: .28rem .9rem;
    font-size: .65rem; font-weight: 700; letter-spacing: .22em;
    text-transform: uppercase; color: var(--accent) !important;
    margin-bottom: 1.1rem;
    font-family: var(--head) !important;
}
.badge::before { content: '⬤'; font-size: .45rem; }
.hero-title {
    font-family: var(--head) !important;
    font-size: clamp(2rem,5vw,2.9rem) !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    background: linear-gradient(140deg,#E2EAF4 30%,#4A90F5 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: .7rem !important;
}
.hero-sub {
    color: var(--muted) !important;
    font-size: .95rem !important;
    line-height: 1.65 !important;
    max-width: 460px !important;
    margin: 0 auto !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] section {
    background: var(--s1) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--r) !important;
    padding: 1.5rem !important;
    transition: border-color .2s, background .2s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important;
    background: rgba(74,144,245,.04) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--muted) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: var(--muted) !important;
    font-family: var(--font) !important;
}
/* Browse button inside uploader */
[data-testid="stFileUploader"] button {
    background: rgba(74,144,245,.15) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] button:hover {
    background: rgba(74,144,245,.25) !important;
}
/* Uploaded filename tag */
[data-testid="stFileUploaderFile"] {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderFile"] * { color: var(--text) !important; }

/* ── Cards ── */
.card {
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.25rem 1.4rem;
    margin-bottom: .9rem;
}
.card-lbl {
    font-family: var(--head) !important;
    font-size: .62rem; font-weight: 700;
    letter-spacing: .2em; text-transform: uppercase;
    color: var(--muted) !important;
    margin-bottom: 1rem;
}

/* ── Winner box ── */
.winner {
    background: linear-gradient(135deg, var(--s2) 0%, rgba(74,144,245,.08) 100%);
    border: 1px solid rgba(74,144,245,.3);
    border-radius: var(--r);
    padding: 1.75rem 1.5rem;
    text-align: center;
    margin-bottom: .9rem;
}
.w-emoji { font-size: 2.8rem; display: block; margin-bottom: .4rem; }
.w-age   {
    font-family: var(--head) !important;
    font-size: 2rem; font-weight: 800;
    color: var(--text) !important;
    margin-bottom: .2rem;
}
.w-conf  { font-size: .92rem; color: var(--accent) !important; font-weight: 500; }

/* ── Top-3 ── */
.t3 { display: flex; gap: .65rem; }
.t3c {
    flex: 1; background: var(--s2); border-radius: 10px;
    padding: .9rem .6rem; text-align: center;
}
.t3e { font-size: 1.5rem; }
.t3a { font-family: var(--head) !important; font-size: .88rem; font-weight: 800; margin: .2rem 0 .1rem; color: var(--text) !important; }
.t3p { font-size: .76rem; font-weight: 600; }
.t3r { font-size: .62rem; color: var(--muted) !important; margin-top: .1rem; }

/* ── Bars ── */
.bars { display: flex; flex-direction: column; gap: .5rem; }
.brow { display: flex; align-items: center; gap: .65rem; }
.blbl { width: 50px; font-size: .76rem; color: var(--muted) !important; text-align: right; flex-shrink: 0; }
.btrk { flex: 1; height: 7px; background: var(--s2); border-radius: 999px; overflow: hidden; }
.bfil { height: 100%; border-radius: 999px; transition: width .5s cubic-bezier(.34,1.56,.64,1); }
.bval { width: 40px; font-size: .76rem; font-weight: 500; color: var(--text) !important; flex-shrink: 0; }
.brow.top .blbl { color: var(--text) !important; font-weight: 600; }
.brow.top .bval { color: var(--accent) !important; font-weight: 700; }

/* ── Divider ── */
.div { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── Warning / info banners ── */
.warn {
    background: rgba(245,158,11,.1);
    border: 1px solid rgba(245,158,11,.35);
    border-left: 4px solid var(--amber);
    border-radius: var(--r); padding: 1rem 1.1rem;
    margin-bottom: .9rem;
}
.warn-title { font-family: var(--head) !important; font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--amber) !important; margin-bottom: .4rem; }
.warn p { color: var(--text) !important; font-size: .88rem; line-height: 1.6; }
.warn code { background: rgba(255,255,255,.08); padding: .1rem .35rem; border-radius: 4px; font-size: .82rem; color: var(--amber) !important; }

.noface {
    background: rgba(239,68,68,.08);
    border: 1px solid rgba(239,68,68,.3);
    border-left: 4px solid var(--red);
    border-radius: var(--r); padding: 1rem 1.1rem;
    margin-bottom: .9rem;
}
.noface-title { font-family: var(--head) !important; font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--red) !important; margin-bottom: .3rem; }
.noface p { color: var(--text) !important; font-size: .88rem; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--s1) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    margin-top: .7rem !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] > div:first-child {
    border-bottom: 1px solid var(--border) !important;
    padding: .8rem 1rem !important;
    background: transparent !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--head) !important;
    font-size: .72rem !important; font-weight: 700 !important;
    letter-spacing: .14em !important; text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--accent) !important; }
[data-testid="stExpander"] > div:last-child {
    padding: 1rem 1.1rem !important;
    background: transparent !important;
}
/* Remove ALL white/grey overlays inside expanders */
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] > div {
    background: transparent !important;
}
[data-testid="stExpander"] p  { color: var(--muted) !important; font-size: .87rem !important; line-height: 1.65 !important; }
[data-testid="stExpander"] li { color: var(--muted) !important; font-size: .87rem !important; }
[data-testid="stExpander"] strong { color: var(--text) !important; }
[data-testid="stExpander"] a  { color: var(--accent) !important; }
[data-testid="stExpander"] code {
    background: rgba(255,255,255,.06) !important;
    color: var(--amber) !important;
    padding: .1rem .3rem !important; border-radius: 4px !important;
}

/* ── Raw probs rows inside expander ── */
.prob-row {
    display: flex; justify-content: space-between;
    padding: .35rem 0; border-bottom: 1px solid rgba(31,48,80,.6);
    font-size: .85rem;
}
.prob-row:last-child { border-bottom: none; }
.prob-name { color: var(--muted) !important; }
.prob-val  { color: var(--text) !important; font-weight: 500; }
.prob-val.top { color: var(--accent) !important; font-weight: 700; }

/* ── Empty state ── */
.empty {
    text-align: center; padding: 3.5rem 1rem 2.5rem;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: .6; }
.empty-title {
    font-family: var(--head) !important;
    font-size: 1.05rem; font-weight: 800;
    color: var(--text) !important; margin-bottom: .5rem;
}
.empty-sub { color: var(--muted) !important; font-size: .87rem; line-height: 1.7; max-width: 340px; margin: 0 auto; }

/* ── Footer ── */
.ft {
    text-align: center; color: var(--muted) !important;
    font-size: .74rem; line-height: 2.1;
    padding: 2rem 0 .5rem; border-top: 1px solid var(--border);
    margin-top: 2rem;
}
.ft strong { color: var(--text) !important; }
.ft a { color: var(--accent) !important; text-decoration: none; }

/* ── Streamlit image widget ── */
[data-testid="stImage"] { text-align: center; }
[data-testid="stImage"] img {
    border-radius: var(--r) !important;
    border: 1px solid var(--border) !important;
    max-height: 340px !important;
    object-fit: cover !important;
    width: 100% !important;
}
[data-testid="stImage"] p { color: var(--muted) !important; font-size: .78rem !important; margin-top: .4rem !important; }

/* ── Column gaps ── */
[data-testid="column"] { padding: 0 .5rem !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ── Alert ── */
[data-testid="stAlert"] {
    background: var(--s1) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}
[data-testid="stAlert"] p { color: var(--text) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
"""

# ── HTML helpers ──────────────────────────────────────────────────────────────
def h_winner(age, conf, emoji):
    return (f'<div class="winner">'
            f'<span class="w-emoji">{emoji}</span>'
            f'<div class="w-age">Age {age}</div>'
            f'<div class="w-conf">{conf:.1f}% confidence</div>'
            f'</div>')

def h_top3(results):
    top3 = sorted(results, key=lambda x: x[1], reverse=True)[:3]
    inner = ""
    for rank,(age,conf) in enumerate(top3):
        i = AGE_NAMES.index(age)
        inner += (f'<div class="t3c" style="border-top:3px solid {AGE_COLOR[i]};'
                  f'opacity:{1-rank*0.18:.2f};">'
                  f'<div class="t3e">{AGE_EMOJI[i]}</div>'
                  f'<div class="t3a">{age}</div>'
                  f'<div class="t3p" style="color:{AGE_COLOR[i]}">{conf:.1f}%</div>'
                  f'<div class="t3r">#{rank+1}</div></div>')
    return (f'<div class="card"><div class="card-lbl">Top 3 Predictions</div>'
            f'<div class="t3">{inner}</div></div>')

def h_bars(results):
    top_conf = max(c for _,c in results)
    rows = ""
    for i,(age,conf) in enumerate(results):
        is_top = abs(conf - top_conf) < 0.01
        pct    = conf / top_conf * 100 if top_conf > 0 else 0
        cls    = "brow top" if is_top else "brow"
        rows += (f'<div class="{cls}">'
                 f'<span class="blbl">{age}</span>'
                 f'<div class="btrk"><div class="bfil" '
                 f'style="width:{pct:.1f}%;background:{AGE_COLOR[i]};"></div></div>'
                 f'<span class="bval">{conf:.1f}%</span></div>')
    return (f'<div class="card"><div class="card-lbl">Full Probability Distribution</div>'
            f'<div class="bars">{rows}</div></div>')

def h_warn_undertrained():
    return """
    <div class="warn">
        <div class="warn-title">⚠️ Model Weights Not Fully Trained</div>
        <p>The loaded <code>fairface_cnn_recommended.pt</code> file contains
        near-initialisation weights — the model has not yet learned to distinguish
        age groups, so all predictions show ~11% confidence (random chance for 9 classes).</p>
        <p style="margin-top:.6rem;">
        <strong>To fix this:</strong> In your Jupyter Notebook, run the full training loop
        to completion, then run the export cell at the bottom:<br>
        <code>torch.save(model_mit2.state_dict(), "fairface_cnn_recommended.pt")</code><br>
        Make sure this cell runs <em>after</em> the training loop finishes and the best
        checkpoint is loaded back with
        <code>model_mit2.load_state_dict(torch.load(best_mit2_path))</code>.
        </p>
    </div>"""

def h_noface():
    return """
    <div class="noface">
        <div class="noface-title">⚠️ No Face Detected</div>
        <p>This image does not appear to contain a human face.
        Please upload a clear face photo for accurate age group prediction.</p>
    </div>"""

def h_empty():
    return """
    <div class="empty">
        <div class="empty-icon">📷</div>
        <div class="empty-title">No image uploaded yet</div>
        <div class="empty-sub">
            Drag and drop or click the upload area above.<br>
            Supported: JPG · PNG · WEBP · BMP · GIF · TIFF
        </div>
    </div>"""

def h_prob_rows(results):
    top_conf = max(c for _,c in results)
    rows = ""
    for age,conf in results:
        is_top = abs(conf - top_conf) < 0.01
        vc = "prob-val top" if is_top else "prob-val"
        rows += (f'<div class="prob-row">'
                 f'<span class="prob-name">{age}</span>'
                 f'<span class="{vc}">{conf:.2f}%</span></div>')
    return f'<div style="margin-top:.4rem;">{rows}</div>'

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="FairFace Age Classifier",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="badge">CAME Assignment 1 &nbsp;·&nbsp; FairFace CNN</div>
        <div class="hero-title">Age Group Classifier</div>
        <div class="hero-sub">
            Upload any face photo and a CNN trained from scratch on the FairFace
            dataset predicts the age group — with full probability breakdown
            across all 9 classes.
        </div>
    </div>""", unsafe_allow_html=True)

    # File uploader — wide accept to handle all formats
    uploaded = st.file_uploader(
        label="Drop an image or click to browse",
        type=["jpg","jpeg","png","webp","bmp","tiff","tif","gif","heic"],
        help="JPG · PNG · WEBP · BMP · TIFF · GIF — any resolution",
        label_visibility="collapsed",
    )

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    if uploaded is not None:
        # Open image robustly
        try:
            raw = Image.open(uploaded)
            if getattr(raw, "n_frames", 1) > 1:
                raw.seek(0)
            pil_img = raw.convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}. Please try a different file.")
            st.stop()

        # Face heuristic check
        has_face, skin_ratio = looks_like_face(pil_img)

        # Run inference
        with st.spinner("Analysing…"):
            results, status = predict(pil_img)

        # Model health warnings (shown above results, not blocking)
        if status == "weights_missing":
            st.error(f"Model file not found at: `{MODEL_PATH}`")
            st.stop()
        elif status.startswith("error:"):
            st.error(f"Model load error: {status[6:]}")
            st.stop()
        elif status == "undertrained":
            st.markdown(h_warn_undertrained(), unsafe_allow_html=True)

        # Face detection warning
        if not has_face:
            st.markdown(h_noface(), unsafe_allow_html=True)

        if results is None:
            st.stop()

        # Determine top result
        top_age, top_conf = max(results, key=lambda x: x[1])
        top_emoji = AGE_EMOJI[AGE_NAMES.index(top_age)]

        # Layout: image | winner + top3
        col_l, col_r = st.columns([1, 1], gap="medium")
        with col_l:
            st.image(pil_img, use_column_width=True, caption="Uploaded image")
        with col_r:
            st.markdown(h_winner(top_age, top_conf, top_emoji), unsafe_allow_html=True)
            st.markdown(h_top3(results), unsafe_allow_html=True)

        # Full bar chart
        st.markdown(h_bars(results), unsafe_allow_html=True)

        # Raw probabilities expander
        with st.expander("📊  All class probabilities"):
            st.markdown(h_prob_rows(results), unsafe_allow_html=True)

    else:
        st.markdown(h_empty(), unsafe_allow_html=True)

    # About expander
    with st.expander("ℹ️  About this model"):
        st.markdown("""
**What it does**  
Classifies a face image into one of 9 age groups: `0-2`, `3-9`, `10-19`, `20-29`,
`30-39`, `40-49`, `50-59`, `60-69`, `70+`.

**Architecture**  
5-block CNN (Conv → BatchNorm → ReLU → MaxPool) with a **residual skip connection**
on block 4→5, Global Average Pooling, and a 2-layer head (512 → 128 → 9).
Trained **from scratch** in PyTorch — no pretrained weights used.

**Dataset**  
[FairFace](https://huggingface.co/datasets/HuggingFaceM4/FairFace) (config `0.25`),
stratified 20,000-sample subset. Input images resized to 64 × 64.

**Fairness**  
Audited across 7 race groups and 2 gender groups.
Trained with **Balanced Mini-Batch Sampling** to reduce race accuracy gap
from 8.14 pp → 6.28 pp. Overall test accuracy: ~41.85%.

**⚠️ Limitations & Responsible Use**  
- Research prototype only — not a production system.  
- Lower performance on age extremes (infants, elderly).  
- Must **not** be used for surveillance, identity verification, or any
  consequential decision-making without ethical and legal review
  (PDPA Sri Lanka / GDPR EU).
""")

    # Footer
    st.markdown("""
    <div class="ft">
        <strong>FairFace CNN Age Group Classifier</strong><br>
        CAME Individual Assignment 1 &nbsp;·&nbsp;
        <a href="https://huggingface.co/datasets/HuggingFaceM4/FairFace" target="_blank">FairFace dataset</a>
        (config 0.25) &nbsp;·&nbsp; PyTorch from scratch &nbsp;·&nbsp;
        Balanced Mini-Batch Sampling<br>
        <span style="color:#4A90F5">Institute of Java and Software Engineering (IJSE)</span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
