# Deploying the app

The app is built to deploy on **Streamlit Community Cloud** with no build step:
everything it reads is committed under `data/processed/app/` and `models/`
(~16 MB total), so the cloud instance trains nothing.

**Live:** <https://ballpark-mkmljvquubqdhwezbkgdtg.streamlit.app/>

## One-time setup (done)

1. Push this repo to GitHub (`AbirChakraborty1/ballpark`).
2. Go to <https://share.streamlit.io> → **New app**.
3. Repository `AbirChakraborty1/ballpark`, branch `main`, main file `app/Home.py`.
4. **Deploy.** First boot takes ~2 minutes to install `requirements.txt`.
5. The URL is wired into `README.md`, `reports/report.md` and
   `reports/outreach_draft.md`. Streamlit Cloud redeploys on every push to `main`.

## If the build runs out of memory

The default 1 GB instance is enough. If it ever isn't, the heaviest object is
`models/winprob.joblib` (~13 MB); the Tactics page is the only one that loads
it, and it does so lazily via `st.cache_resource`.

## Refreshing after a new cricsheet dump

```bash
python scripts/run_all.py         # rebuilds everything incl. the app bundle
git add -f data/processed/app models/outcome.joblib models/winprob.joblib
git commit -m "refresh: cricsheet dump <date>"
git push
```

Streamlit Cloud redeploys automatically on push.
