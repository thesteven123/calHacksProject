import { describe, expect, it } from 'vitest'
import { buildDetection } from './fixtures'

// API_CONTRACT fusion: fake if any specialist says fake; real only if all say real; else review.
describe('mock fusion rule', () => {
  it('is fake when any specialist is fake', () => {
    const d = buildDetection({ id: 'x', mediaType: 'image', general: 0.1, face: { p: 0.95, box: [0, 0, 1, 1], score: 0.9 } })
    expect(d.verdict).toBe('fake')
  })

  it('is real only when all specialists are real', () => {
    const d = buildDetection({ id: 'x', mediaType: 'image', general: 0.1, face: { p: 0.2, box: [0, 0, 1, 1], score: 0.9 } })
    expect(d.verdict).toBe('real')
  })

  it('is review when no fake and at least one review', () => {
    const d = buildDetection({ id: 'x', mediaType: 'image', general: 0.1, face: { p: 0.6, box: [0, 0, 1, 1], score: 0.9 } })
    expect(d.verdict).toBe('review')
  })
})
