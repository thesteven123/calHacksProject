import { useRef, useState } from 'react'

interface Props {
  onClose: () => void
  onSubmit: (file: File, caption: string) => void
  dark?: boolean
}

export const ACCEPT = 'image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/webm'

// Picks a file and caption, then hands both to the feed for a live scan.
export function UploadSheet({ onClose, onSubmit, dark = false }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [caption, setCaption] = useState('')
  const input = useRef<HTMLInputElement>(null)
  const surface = dark ? 'bg-neutral-900 text-neutral-100' : 'bg-white text-neutral-900'
  const field = dark ? 'border-neutral-700 bg-neutral-800' : 'border-neutral-300 bg-white'

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 sm:items-center" onClick={onClose}>
      <form
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-title"
        onClick={(e) => e.stopPropagation()}
        onSubmit={(e) => {
          e.preventDefault()
          if (file) onSubmit(file, caption)
        }}
        className={`w-full max-w-md space-y-4 rounded-t-2xl p-5 shadow-2xl sm:rounded-2xl ${surface}`}
      >
        <h2 id="upload-title" className="text-lg font-semibold">
          Scan a new post
        </h2>
        <input ref={input} type="file" accept={ACCEPT} className="sr-only" id="upload-file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <label
          htmlFor="upload-file"
          className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-sm focus-within:ring-2 focus-within:ring-sky-400 ${field}`}
        >
          {file ? file.name : 'Choose an image (JPG, PNG, WebP) or video (MP4, MOV, WebM)'}
        </label>
        <input
          type="text"
          value={caption}
          onChange={(e) => setCaption(e.target.value)}
          placeholder="Caption"
          maxLength={300}
          aria-label="Caption"
          className={`w-full rounded-lg border px-3 py-2 text-sm ${field}`}
        />
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="rounded-lg px-4 py-2 text-sm underline">
            Cancel
          </button>
          <button
            type="submit"
            disabled={!file}
            className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
          >
            Post
          </button>
        </div>
      </form>
    </div>
  )
}
