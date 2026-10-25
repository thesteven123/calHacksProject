# DFD Gradio demo

A Gradio front end for the detector. It calls a hosted detector's `POST /v1/detect`, or runs a labelled mock when no backend is set.

## Run locally

```
cd demo/gradio
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python app.py                       # mock mode (no model)
DFD_API_URL=https://<host> DFD_API_KEY=<key> .venv/bin/python app.py   # real detector
```

Local URL: http://127.0.0.1:7860. Secrets come from the environment only, never from the repo.

## Deploy to Vercel

Import the repo in Vercel with **Root Directory** set to `demo/gradio`. Vercel installs `requirements.txt` and serves `api/index.py`, which mounts the Gradio UI at `/`.
Set `DFD_API_URL` and `DFD_API_KEY` in the Vercel project settings.

Limits to know before relying on this:
- Gradio is not a native Vercel framework. This is an ASGI mount, so it is untested on Vercel until deployed.
- Vercel caps request bodies (about 4.5 MB), so large uploads (images up to 15 MB, videos up to 50 MB) will fail there. Use a Hugging Face Space or the Modal service for full-size files.
- Functions have a time limit (`maxDuration` in `vercel.json`); a cold start plus model call must fit inside it.
- The Gradio bundle is large. If the Vercel build exceeds the function size limit, host on Hugging Face Spaces instead.

## Badges and copy

The UI shows the API's `label`, `verdict`, `confidence`, `reasons`, and specialist scores. It adds no claims beyond these.
