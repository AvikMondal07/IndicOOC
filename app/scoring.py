"""
IndicOOC scoring module.

The zero-shot SigLIP scoring logic from IndicOOC_01_ZeroShot_Baseline.ipynb,
pulled out into plain functions so both the notebook and the Streamlit app
(app/app.py) call the same code instead of drifting apart.
"""

from io import BytesIO
from typing import Optional, Tuple

import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip-base-patch16-224"
DEFAULT_THRESHOLD = 0.5

_model = None
_processor = None
_device = None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load SigLIP once and cache it at module level.

    In app.py this is wrapped in @st.cache_resource so Streamlit only
    actually calls it once per server session, not on every interaction.
    """
    global _model, _processor, _device
    if _model is None:
        _device = get_device()
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
    return _model, _processor, _device


def load_image_from_url(url: str, timeout: int = 15) -> Optional[Image.Image]:
    """Same graceful-failure loader as the baseline notebook — returns
    None on any download/decode failure instead of raising, given the
    CDN link-rot risk already documented in pilot_v0.csv."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def score_pair(image: Image.Image, claim_text: str) -> float:
    """SigLIP's sigmoid match probability for one (image, claim) pair."""
    model, processor, device = load_model()
    inputs = processor(
        text=[claim_text], images=image, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        score = torch.sigmoid(outputs.logits_per_image)[0][0].item()
    return score


def classify(score: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Threshold a match score into two of IndicOOC's three classes.

    Deliberately does NOT implement 'insufficient_evidence' — a bare
    similarity score can't express "not enough information," that needs
    the O3 caption-grounded model, not zero-shot thresholding. Flagging
    this explicitly so it isn't silently missing from the demo.
    """
    return "supported" if score > threshold else "out_of_context"


def score_and_classify(
    image: Image.Image, claim_text: str, threshold: float = DEFAULT_THRESHOLD
) -> Tuple[str, float]:
    score = score_pair(image, claim_text)
    return classify(score, threshold), score
