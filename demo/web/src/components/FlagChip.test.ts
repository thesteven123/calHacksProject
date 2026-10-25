import { describe, expect, it } from 'vitest'
import { buildDetection } from '../mock/fixtures'
import { badgeFor } from './FlagChip'

describe('badgeFor', () => {
  it('shows no badge for a real verdict', () => {
    expect(badgeFor(buildDetection({ id: 'r', mediaType: 'image', general: 0.05 }))).toBeNull()
  })

  it('shows neutral Unverified for review, even at high confidence', () => {
    const d = buildDetection({ id: 'v', mediaType: 'image', general: 0.85 })
    expect(d.verdict).toBe('review')
    expect(badgeFor(d)).toEqual({ kind: 'review', text: 'Unverified' })
  })

  it('shows Suspected Deepfake for fake at confidence >= 0.8', () => {
    const d = buildDetection({ id: 'f', mediaType: 'image', general: 0.97 })
    expect(d.verdict).toBe('fake')
    expect(badgeFor(d)).toEqual({ kind: 'suspected', text: 'Suspected Deepfake' })
  })

  it('shows Altered with AI for fake below 0.8 confidence', () => {
    // Real API thresholds are calibrated; the mock cannot produce this case, so set it directly.
    const d = { ...buildDetection({ id: 'a', mediaType: 'image', general: 0.95 }), verdict: 'fake' as const, confidence: 0.6 }
    expect(badgeFor(d)).toEqual({ kind: 'altered', text: 'Altered with AI' })
  })
})
