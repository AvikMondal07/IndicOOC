"""
IndicOOC — Streamlit MVP (Review 1 target)

Two ways to try the zero-shot SigLIP baseline (Objective O2-i):
  1. Upload your own image + type a claim
  2. Pick one of the 10 pilot_v0.csv rows and see the model's call vs.
     the ground-truth label (good for a live demo — no internet-dependent
     upload needed mid-review)

Run locally:
    streamlit run app/app.py

Share during the review (Colab has no public URL by default):
    pip install pyngrok
    streamlit run app/app.py &
    python -c "from pyngrok import ngrok; print(ngrok.connect(8501))"
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from scoring import DEFAULT_THRESHOLD, load_image_from_url, load_model, score_and_classify

PILOT_CSV = Path(__file__).parent.parent / "data" / "annotations" / "pilot_v0.csv"

st.set_page_config(page_title="IndicOOC — Zero-Shot Demo", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading SigLIP (first run only)…")
def get_model():
    return load_model()


@st.cache_data
def get_pilot_data():
    if not PILOT_CSV.exists():
        return None
    return pd.read_csv(PILOT_CSV)


def render_verdict(label: str, score: float, threshold: float, ground_truth: str = None):
    color = "green" if label == "supported" else "red"
    st.markdown(f"### Model verdict: :{color}[{label.upper()}]")
    st.progress(min(max(score, 0.0), 1.0), text=f"SigLIP match score: {score:.3f} (threshold {threshold:.2f})")
    if ground_truth is not None:
        if ground_truth == label:
            st.success(f"Matches ground truth ({ground_truth}).")
        elif ground_truth not in ("supported", "out_of_context"):
            st.info(
                f"Ground truth is '{ground_truth}' — this baseline only outputs "
                "supported/out_of_context, so it can't express that class. Expected miss."
            )
        else:
            st.warning(f"Ground truth is '{ground_truth}' — model disagrees.")
    st.caption(
        "Note: this zero-shot threshold has no 'insufficient_evidence' class — a bare "
        "similarity score can't express 'not enough information.' That needs the O3 "
        "caption-grounded model, not thresholding. See scoring.py for why."
    )


st.title("🔎 IndicOOC — Zero-Shot Out-of-Context Detector")
st.write(
    "Objective O2-i baseline: SigLIP similarity thresholding on (image, claim) pairs, "
    "for Hindi / Bengali / Marathi / code-mixed claims."
)

# Load model once, up front, so both tabs share it and errors surface immediately.
try:
    get_model()
except Exception as e:
    st.error(f"Could not load SigLIP: {e}")
    st.stop()

threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
st.sidebar.caption(f"Model: `google/siglip-base-patch16-224` (English-trained checkpoint — see Section 5.2 caveat)")

tab_upload, tab_pilot = st.tabs(["📤 Upload your own", "🗂️ Try a pilot example"])

with tab_upload:
    uploaded_file = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
    claim_text = st.text_area("Claim text", placeholder="Type or paste the claim circulating with this image…")
    run_upload = st.button("Check this pair", type="primary", disabled=not (uploaded_file and claim_text))

    if run_upload:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        with st.spinner("Scoring…"):
            label, score = score_and_classify(image, claim_text, threshold)
        render_verdict(label, score, threshold)

with tab_pilot:
    pilot_df = get_pilot_data()
    if pilot_df is None:
        st.warning(
            "pilot_v0.csv not found next to this app (expected at "
            "`data/annotations/pilot_v0.csv`). Upload it or adjust PILOT_CSV in app.py."
        )
    else:
        row_id = st.selectbox("Pick a pilot row", pilot_df["id"].tolist())
        row = pilot_df[pilot_df["id"] == row_id].iloc[0]
        st.write(f"**Claim ({row['language']}):** {row['claim_text']}")
        st.caption(f"Fact-checker: {row['fact_checker']} · Ground-truth label: `{row['label']}`")
        run_pilot = st.button("Run this row", type="primary")

        if run_pilot:
            with st.spinner("Downloading image…"):
                image = load_image_from_url(row["image_url"])
            if image is None:
                st.error("Image failed to download (link rot) — this is the exact failure mode scoring.py logs.")
            else:
                st.image(image, caption=row_id, use_container_width=True)
                with st.spinner("Scoring…"):
                    label, score = score_and_classify(image, row["claim_text"], threshold)
                render_verdict(label, score, threshold, ground_truth=row["label"])
