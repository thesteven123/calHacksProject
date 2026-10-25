import { useEffect, useState } from 'react'
import { api } from '../api'
import type { ModelsResponse } from '../types'
import { AppShell } from '../components/AppShell'

// Public Modal api host (no secret). Used when /api/models is unavailable.
const PUBLIC_API_URL = 'https://thesteven123--dfd-demo-api.us-east.modal.run'

const SAMPLE = `{
  "id": "3f2a9c1e-5b7d-4e8a-9c0f-1a2b3c4d5e6f",
  "media_type": "image",
  "verdict": "fake",
  "flag": "ai_generated",
  "label": "Likely AI-generated or manipulated",
  "confidence": 0.97,
  "specialists": [
    { "name": "general", "title": "General AI-image detector", "applies_to": "whole image",
      "p_fake": 0.12, "verdict": "real", "calibration_status": "matched" },
    { "name": "face", "title": "AI-generated face detector", "applies_to": "face crop",
      "p_fake": 0.97, "verdict": "fake", "calibration_status": "matched", "box": [412, 188, 240, 260] }
  ],
  "faces": [{ "box": [412, 188, 240, 260], "score": 0.98 }],
  "video": null,
  "reasons": ["The face crop scores 0.97 on the AI-face detector (threshold 0.95)."],
  "latency_ms": 412,
  "disclaimer": "Research demo. Not for enforcement decisions; uncertain cases go to human review."
}`

// Copy button that reports success for a moment.
function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false)
  return (
    <button
      type="button"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setDone(true)
          setTimeout(() => setDone(false), 1500)
        })
      }}
      className="rounded-md border border-neutral-700 px-2 py-1 text-xs hover:bg-neutral-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
    >
      {done ? 'Copied' : 'Copy'}
    </button>
  )
}

export function ApiPage() {
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api<ModelsResponse>('/api/models')
      .then(setModels)
      .catch(() => setError('Could not load model details.'))
  }, [])

  const base = models?.api_url || PUBLIC_API_URL
  const curl = `curl -H "Authorization: Bearer $DFD_API_KEY" -F file=@photo.jpg ${base}/v1/detect`

  return (
    <AppShell>
      <main className="min-h-0 flex-1 space-y-8 overflow-y-auto p-4 md:p-8">
        <section className="max-w-3xl space-y-3">
          <h1 className="text-2xl font-semibold">API</h1>
          <p className="text-sm text-neutral-400">
            <code className="rounded bg-neutral-800 px-1.5 py-0.5">POST /v1/detect</code> takes a multipart <code>file</code> field and a
            Bearer key. Images up to 15 MB (JPG, PNG, WebP); videos up to 60 s and 50 MB (MP4, MOV, WebM).
          </p>
          <div className="flex items-start gap-2 rounded-xl bg-neutral-900 p-3 ring-1 ring-neutral-800">
            <pre className="min-w-0 flex-1 overflow-x-auto whitespace-pre-wrap break-all text-xs text-neutral-200">{curl}</pre>
            <CopyButton text={curl} />
          </div>
        </section>

        <section className="max-w-3xl space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Sample response</h2>
            <CopyButton text={SAMPLE} />
          </div>
          <pre className="overflow-x-auto rounded-xl bg-neutral-900 p-3 text-xs text-neutral-200 ring-1 ring-neutral-800">{SAMPLE}</pre>
        </section>

        {error && (
          <p role="alert" className="text-sm text-red-400">
            {error}
          </p>
        )}
        {models && (
          <section className="max-w-3xl space-y-4">
            <div>
              <h2 className="font-semibold">Fusion</h2>
              <p className="text-sm text-neutral-300">{models.fusion}</p>
            </div>
            {models.specialists.map((s) => (
              <div key={s.name} className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800">
                <h3 className="font-medium">{s.title}</h3>
                <p className="text-xs text-neutral-500">Checkpoint {s.checkpoint} · Trained on {s.trained_on}</p>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-neutral-300">
                  {(Array.isArray(s.limitations) ? s.limitations : [s.limitations]).map((l) => (
                    <li key={l}>{l}</li>
                  ))}
                </ul>
              </div>
            ))}
            {models.notes.length > 0 && (
              <div>
                <h2 className="font-semibold">Notes</h2>
                <ul className="list-disc space-y-1 pl-5 text-sm text-neutral-300">
                  {models.notes.map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}
      </main>
    </AppShell>
  )
}
