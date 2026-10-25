import { useCallback, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { useDetect } from '../hooks/useDetect'
import type { Detection } from '../types'
import { AppShell } from '../components/AppShell'
import { MediaWithBoxes } from '../components/MediaWithBoxes'
import { SpecialistRows } from '../components/SpecialistRows'
import { VerdictBadge } from '../components/VerdictBadge'
import { ACCEPT } from '../components/UploadSheet'

type Msg =
  | { id: string; role: 'user'; url: string; mediaType: 'image' | 'video'; name: string }
  | { id: string; role: 'assistant'; detection: Detection; url: string; mediaType: 'image' | 'video' }
  | { id: string; role: 'assistant'; error: string }
  | { id: string; role: 'pending' }

const STRIP: Record<Detection['verdict'], string> = { fake: 'bg-red-500', review: 'bg-amber-400', real: 'bg-emerald-500' }

function DetectionCard({ detection, url }: { detection: Detection; url: string }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const frames = detection.video?.frames ?? []
  return (
    <article className="w-full max-w-3xl space-y-4 rounded-2xl bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <header className="flex flex-wrap items-center gap-3">
        <VerdictBadge verdict={detection.verdict} />
        <h2 className="font-semibold">{detection.label}</h2>
        <span className="text-sm text-neutral-400">Confidence {Math.round(detection.confidence * 100)}%</span>
        <span className="ml-auto text-xs tabular-nums text-neutral-500">{detection.latency_ms} ms</span>
      </header>
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <MediaWithBoxes url={url} mediaType={detection.media_type} faces={detection.faces} verdict={detection.verdict} videoRef={videoRef} />
          {frames.length > 0 && (
            <div className="mt-3">
              <div className="flex gap-1" role="list" aria-label="Keyframes">
                {frames.map((f, i) => (
                  <button
                    key={i}
                    type="button"
                    role="listitem"
                    title={`t=${f.t}s, p_fake ${f.p_fake.toFixed(2)}`}
                    aria-label={`Keyframe at ${f.t} seconds, ${f.verdict}`}
                    onClick={() => {
                      if (videoRef.current) videoRef.current.currentTime = f.t
                    }}
                    className={`h-3 flex-1 rounded-sm ${STRIP[f.verdict]} focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400`}
                  />
                ))}
              </div>
              <p className="mt-1 text-xs text-neutral-500">
                Keyframes are not calibrated. Colour shows each frame's verdict.
              </p>
            </div>
          )}
        </div>
        <div className="space-y-4">
          <SpecialistRows specialists={detection.specialists} />
          {detection.reasons.length > 0 && (
            <ul className="list-disc space-y-1 pl-5 text-sm text-neutral-300">
              {detection.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <footer className="space-y-1 text-xs text-neutral-500">
        <p>{detection.disclaimer}</p>
        <p>Models: {Object.entries(detection.model_versions).map(([k, v]) => `${k} ${v}`).join(', ')}</p>
      </footer>
    </article>
  )
}

export function Playground() {
  const [messages, setMessages] = useState<Msg[]>([])
  const [dragging, setDragging] = useState(false)
  const { run, busy } = useDetect()
  const fileInput = useRef<HTMLInputElement>(null)

  const send = useCallback(
    async (file: File) => {
      const url = URL.createObjectURL(file)
      const mediaType: 'image' | 'video' = file.type.startsWith('video/') ? 'video' : 'image'
      const pendingId = crypto.randomUUID()
      setMessages((m) => [
        ...m,
        { id: crypto.randomUUID(), role: 'user', url, mediaType, name: file.name },
        { id: pendingId, role: 'pending' },
      ])
      const detection = await run(file)
      if (!detection) URL.revokeObjectURL(url)
      setMessages((m) =>
        m.map((x) =>
          x.id !== pendingId
            ? x
            : detection
              ? { id: pendingId, role: 'assistant', detection, url, mediaType }
              : { id: pendingId, role: 'assistant', error: 'Scan failed. Check the file type and size, then try again.' },
        ),
      )
    },
    [run],
  )

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && !busy) void send(file)
  }

  return (
    <AppShell
      actions={
        <button
          type="button"
          onClick={() => setMessages([])}
          className="mb-2 rounded-lg border border-neutral-700 px-3 py-2 text-left text-sm hover:bg-neutral-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
        >
          + New chat
        </button>
      }
    >
      <main
        className="relative flex min-h-0 flex-1 flex-col"
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div className="min-h-0 flex-1 space-y-6 overflow-y-auto p-4 md:p-8" aria-live="polite">
          {messages.length === 0 && (
            <div className="mx-auto mt-16 max-w-md text-center">
              <h1 className="text-2xl font-semibold">Is this image or video AI-generated?</h1>
              <p className="mt-2 text-sm text-neutral-400">
                Drop a photo or short clip here, or use the attach button. You get a verdict, the evidence from each detector, and what the
                detector cannot tell you.
              </p>
            </div>
          )}
          {messages.map((m) => {
            if (m.role === 'user')
              return (
                <div key={m.id} className="flex justify-end">
                  <div className="max-w-xs overflow-hidden rounded-2xl bg-neutral-800 p-2">
                    {m.mediaType === 'video' ? (
                      <video src={m.url} muted playsInline className="max-h-72 rounded-xl" />
                    ) : (
                      <img src={m.url} alt={`Your upload: ${m.name}`} className="max-h-72 rounded-xl" />
                    )}
                    <p className="px-1 pt-1 text-xs text-neutral-400">{m.name}</p>
                  </div>
                </div>
              )
            if (m.role === 'pending')
              return (
                <div key={m.id} className="flex items-center gap-2 text-sm text-neutral-400" role="status">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-sky-400" />
                  Analysing... this can take a few seconds.
                </div>
              )
            if ('error' in m) return <p key={m.id} role="alert" className="text-sm text-red-400">{m.error}</p>
            return <DetectionCard key={m.id} detection={m.detection} url={m.url} />
          })}
        </div>
        <div className="border-t border-neutral-800 p-3">
          <div
            className={`mx-auto flex max-w-3xl items-center gap-3 rounded-2xl border bg-neutral-900 px-3 py-2 ${
              dragging ? 'border-sky-400' : 'border-neutral-700'
            }`}
          >
            <input
              ref={fileInput}
              id="playground-file"
              type="file"
              accept={ACCEPT}
              className="sr-only"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) void send(file)
                e.target.value = ''
              }}
            />
            <label
              htmlFor="playground-file"
              className={`cursor-pointer rounded-lg px-3 py-1.5 text-sm hover:bg-neutral-800 focus-within:ring-2 focus-within:ring-sky-400 ${busy ? 'pointer-events-none opacity-40' : ''}`}
            >
              Attach
            </label>
            <p className="flex-1 truncate text-sm text-neutral-400">
              {dragging ? 'Drop to scan' : 'Image (JPG, PNG, WebP up to 15 MB) or video (MP4, MOV, WebM up to 60 s, 50 MB)'}
            </p>
          </div>
        </div>
      </main>
    </AppShell>
  )
}
