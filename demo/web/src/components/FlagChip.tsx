import type { Detection } from '../types'

export interface Badge {
  kind: 'altered' | 'suspected'
  text: string
}

// PRD 5: <50% no badge; 50-79% "Altered with AI"; >=80% on a fake verdict -> "Suspected Deepfake".
export function badgeFor(d: Detection): Badge | null {
  if (d.verdict === 'real') return null
  if (d.verdict === 'fake' && d.confidence >= 0.8) return { kind: 'suspected', text: 'Suspected Deepfake' }
  if (d.confidence >= 0.5) return { kind: 'altered', text: 'Altered with AI' }
  return null
}

interface Props {
  badge: Badge | null
  onOpen?: () => void
  variant: 'inline' | 'overlay'
}

const Sparkle = () => (
  <svg viewBox="0 0 24 24" width="12" height="12" aria-hidden="true" fill="currentColor">
    <path d="M12 2l1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8L12 2zm7 11l.9 2.6L22 16.5l-2.1.9L19 20l-.9-2.6L16 16.5l2.1-.9L19 13z" />
  </svg>
)

const Shield = () => (
  <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 3l8 3v6c0 4.5-3.4 8-8 9-4.6-1-8-4.5-8-9V6l8-3z" />
  </svg>
)

// Home/Explore uses the inline blue tag; Reels uses the shield overlay with a drop shadow (no pill).
export function FlagChip({ badge, onOpen, variant }: Props) {
  if (!badge) return null
  if (variant === 'overlay') {
    return (
      <button
        type="button"
        onClick={onOpen}
        aria-label={`${badge.text}: open forensic details`}
        className="flex items-center gap-1.5 text-sm font-semibold text-white drop-shadow-[0_1px_3px_rgba(0,0,0,0.9)] focus:outline-none focus-visible:ring-2 focus-visible:ring-white"
      >
        <Shield />
        {badge.text}
      </button>
    )
  }
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label={`${badge.text}: open forensic details`}
      className="inline-flex items-center gap-1 rounded-md bg-[#0095F6]/15 px-2 py-0.5 text-xs font-medium text-[#0095F6] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#0095F6]"
    >
      <Sparkle />
      {badge.text}
    </button>
  )
}
