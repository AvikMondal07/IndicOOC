# IndicOOC

Detecting Out-of-Context Image–Claim Pairs in Indic-Language and Code-Mixed Social Media Posts.
M.Tech major project — AI & Data Science, KIIT, Bhubaneswar.

## Status
Review 0 synopsis submitted. Zero-shot SigLIP baseline (Objective O2-i) running on a
10-row pilot set (`data/annotations/pilot_v0.csv`). Streamlit MVP demo in `app/`.

## Repo layout
```
app/
  scoring.py     # shared scoring logic — imported by both app.py and the notebooks
  app.py         # Streamlit MVP: upload an image+claim, or run a pilot example
data/
  annotations/pilot_v0.csv
notebooks/
  IndicOOC_00_Setup.ipynb           # environment sanity check
  IndicOOC_01_ZeroShot_Baseline.ipynb  # pilot-set baseline + metrics
```

## Run the app locally
```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

## Run on Colab (free T4) and share a link for the review
```bash
!pip install -r requirements.txt pyngrok -q
!streamlit run app/app.py &>/content/logs.txt &
from pyngrok import ngrok
print(ngrok.connect(8501))
```
ngrok needs a free account + auth token (`ngrok.set_auth_token(...)`) for a stable link.
localtunnel (`npx localtunnel --port 8501`) is a no-signup alternative.

## Git
```bash
git init
git add .
git commit -m "Zero-shot SigLIP baseline + Streamlit MVP"
git remote add origin <your-repo-url>
git push -u origin main
```
Review 2 wants commit *history* — commit as you go (pilot data, baseline notebook, app,
each dataset-size milestone) rather than one squashed commit before the review.
