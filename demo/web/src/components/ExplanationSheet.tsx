import { useEffect, useRef, useState } from 'react'
import type { Detection, Verdict } from '../types'
import { SpecialistRows } from './SpecialistRows'

interface Props {
  detection: Detection
  onClose: () => void
}

const HEADLINE: Record<Verdict, string> = {
  fake: 'AI or synthetic media suspected',
  review: 'Unverified: needs human review',
  real: 'No manipulation detected',
}

// Forensic sheet: only states what the detector returned (verdict, reasons, specialist scores).
export function ExplanationSheet({ detection, onClose }: Props) {
  const panel = useRef<HTMLDivElement>(null)
  const [note, setNote] = useState<string | null>(null)

  useEffect(() => {
    panel.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const controls = [
    { label: 'Why am I seeing this?', action: `Shown because the detector returned verdict "${detection.verdict}".` },
    {
      label: 'How AI content is identified',
      action: 'Each specialist scores the image or keyframes. Fake if any specialist says fake; real only if all say real; otherwise review.',
    },
    { label: 'Request a human review', action: 'Review requested. A moderator will look at this (demo only, nothing sent).' },
    { label: 'Report incorrect label', action: 'Thanks. This report is recorded in the demo only.' },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60" onClick={onClose}>
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby="explain-title"
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
        className="max-h-[85dvh] w-full max-w-[430px] overflow-y-auto rounded-t-2xl bg-[#1c1b1b] p-5 text-[#fafafa] shadow-2xl focus:outline-none"
      >
        <div className="mx-auto mb-3 h-1 w-10 rounded-full bg-[#262626]" aria-hidden="true" />
        <p className="text-[13px] text-[#8e8e8e]">About this content • Research detector</p>
        <h2 id="explain-title" className="mt-1 text-[14px] font-medium">
          {HEADLINE[detection.verdict]}
        </h2>

        <section className="mt-4 space-y-3">
          <h3 className="text-[14px] font-medium">Detection signals</h3>
          {detection.reasons.length > 0 && (
            <ul className="list-disc space-y-1 rounded-xl bg-[#121212] p-3 pl-8 text-[13px] text-[#d4d4d4]">
              {detection.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}
          <div className="rounded-xl bg-[#121212] p-3">
            <SpecialistRows specialists={detection.specialists} />
          </div>
          {detection.video && (
            <div className="rounded-xl bg-[#121212] p-3 text-[13px]">
              <p className="font-medium">Video method</p>
              <p className="text-[#8e8e8e]">
                {detection.video.method} · {detection.video.n_keyframes} keyframes · calibrated: {detection.video.calibrated ? 'yes' : 'no'}
              </p>
            </div>
          )}
        </section>

        <section className="mt-5 divide-y divide-[#262626] rounded-xl bg-[#121212]">
          {controls.map((c) => (
            <button
              key={c.label}
              type="button"
              onClick={() => setNote(c.action)}
              className="block w-full px-4 py-3 text-left text-[14px] text-[#0095F6] hover:bg-[#1c1b1b] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#0095F6]"
            >
              {c.label}
            </button>
          ))}
        </section>
        {note && (
          <p role="status" className="mt-3 text-[13px] text-[#8e8e8e]">
            {note}
          </p>
        )}
        <p className="mt-5 text-[13px] text-[#8e8e8e]">{detection.disclaimer}</p>
        <button type="button" onClick={onClose} className="mt-4 w-full rounded-lg bg-[#262626] py-2 text-[14px] font-medium">
          Close
        </button>
      </div>
    </div>
  )
}
