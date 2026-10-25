import { useEffect, useRef, useState } from 'react'
import type { Detection } from '../types'
import { SpecialistRows } from './SpecialistRows'

interface Props {
  detection: Detection
  onClose: () => void
}

// PRD 3.3 forensic sheet: grab handle, plain-language signals, and the four user controls.
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

  const face = detection.specialists.find((s) => s.name === 'face')
  const controls = [
    { label: 'Why am I seeing this?', action: 'Shown because the detector scored this content above the label threshold.' },
    { label: 'How AI content is identified', action: 'Detectors score pixels and faces; no single score decides on its own.' },
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
        <p className="text-[13px] text-[#8e8e8e]">About this content • Meta AI Info</p>
        <h2 id="explain-title" className="mt-1 text-[14px] font-medium">
          AI or synthetic media suspected
        </h2>

        <section className="mt-4 space-y-3">
          <h3 className="text-[14px] font-medium">Detection signals</h3>
          <div className="rounded-xl bg-[#121212] p-3">
            <p className="text-[13px] font-medium">Visual patterns</p>
            <p className="mb-2 text-[13px] text-[#8e8e8e]">
              Facial boundary warping and lighting inconsistencies{face ? ` (face score ${Math.round(face.p_fake * 100)}%)` : ''}.
            </p>
            <SpecialistRows specialists={detection.specialists} />
          </div>
          <div className="rounded-xl bg-[#121212] p-3 text-[13px]">
            <p className="font-medium">Audio synthesis</p>
            <p className="text-[#8e8e8e]">Not analysed in this demo.</p>
          </div>
          <div className="rounded-xl bg-[#121212] p-3 text-[13px]">
            <p className="font-medium">Content provenance (C2PA)</p>
            <p className="text-[#8e8e8e]">No C2PA manifest was found for this file.</p>
          </div>
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
