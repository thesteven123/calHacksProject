# DFD demo web

Research demo UI for the deepfake detector: a playground (upload a file, get a verdict), a Spectra Guard feed (`/instagram`), a Reels feed (`/reels`), and an API page (`/api`). Stack: Vite, React, TypeScript, Tailwind, react-router.

The UI only states what the detector returns: the verdict, the API's own `reasons`, and specialist scores. It is not for enforcement decisions.

## Dev

```
npm install
VITE_MOCK=1 npm run dev     # mock mode: fixtures and a random detector, passcode "demo"
npm run dev                 # real mode: proxies /api and /v1 to 127.0.0.1:8000
```

Mock mode serves `/api/*` from `src/mock/` (no backend). Placeholder media is synthetic and comes from `scripts/make-placeholders.sh` (ffmpeg).

## Build

```
npm run build               # tsc --noEmit, then vite build -> dist/
npm run preview             # serve dist locally
```

Asset URLs are relative (`base: './'`), so `dist/` works when served from `/`. The deploy path bundles `dist/` (see `serve/modal_app.py`).

## Layout

- `src/pages/`: `Login`, `Playground`, `ApiPage`, `Instagram` (Spectra Guard feed and profile), `Reels`.
- `src/components/`: `FlagChip` (`badgeFor` rules), `ExplanationSheet` (forensic sheet), `SpecialistRows`, `UploadSheet`, `MediaWithBoxes`.
- `src/hooks/useDetect.ts`: the single upload hook (`POST /api/detect`).
- `src/api.ts`: same-origin fetch wrapper, switching to mock in `VITE_MOCK=1`.

## Badge rules

- `fake` and confidence >= 0.8: "Suspected Deepfake".
- `fake` below 0.8: "Altered with AI".
- `review`: neutral "Unverified". Never an AI label.
- `real`: no badge.
- The profile "Suspected AI" pill counts only `fake` verdicts (more than 60% of posts).

## Limits

- Audio and provenance are not analysed by the detector, and the UI does not claim otherwise.
- "Request a human review" and "Report incorrect label" are local demo actions only.
