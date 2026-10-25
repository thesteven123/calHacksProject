import { useState } from 'react'
import type { Detection } from '../types'

interface Props {
  url: string
  mediaType: 'image' | 'video'
  faces: Detection['faces']
  verdict: Detection['verdict']
  videoRef?: React.RefObject<HTMLVideoElement | null>
}

// Image with face boxes scaled from original pixels (natural size), or a video with controls.
export function MediaWithBoxes({ url, mediaType, faces, verdict, videoRef }: Props) {
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null)
  const stroke = verdict === 'fake' ? 'border-red-500' : verdict === 'review' ? 'border-amber-400' : 'border-emerald-500'

  if (mediaType === 'video') {
    return <video ref={videoRef} src={url} controls muted playsInline className="w-full rounded-lg bg-black" />
  }
  return (
    <div className="relative">
      <img
        src={url}
        alt="Uploaded image being checked"
        onLoad={(e) => setDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        className="w-full rounded-lg"
      />
      {dims &&
        faces.map((f, i) => (
          <div
            key={i}
            className={`absolute border-2 ${stroke}`}
            style={{
              left: `${(f.box[0] / dims.w) * 100}%`,
              top: `${(f.box[1] / dims.h) * 100}%`,
              width: `${(f.box[2] / dims.w) * 100}%`,
              height: `${(f.box[3] / dims.h) * 100}%`,
            }}
          >
            <span className="absolute -top-5 left-0 rounded bg-black/70 px-1 text-[10px] text-white">
              face {Math.round(f.score * 100)}%
            </span>
          </div>
        ))}
    </div>
  )
}
