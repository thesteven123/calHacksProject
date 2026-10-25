import { describe, expect, it } from 'vitest'
import { IG_POSTS } from '../mock/fixtures'
import type { PostView } from '../types'
import { flaggedShare, truthTag } from './feedShared'

const withHandle = (handle: string, post: PostView): PostView => ({ ...post, handle })

describe('flaggedShare', () => {
  it('counts only fake verdicts, not review', () => {
    const fake = IG_POSTS.find((p) => p.detection.verdict === 'fake') as PostView
    const review = IG_POSTS.find((p) => p.detection.verdict === 'review') as PostView
    const posts = [withHandle('acct', fake), withHandle('acct', review)]
    expect(flaggedShare(posts, 'acct')).toEqual({ flagged: 1, total: 2, suspected: false })
  })

  it('marks the account suspected above 60% fake', () => {
    const fake = IG_POSTS.find((p) => p.detection.verdict === 'fake') as PostView
    const posts = [withHandle('acct', fake), withHandle('acct', fake), withHandle('acct', fake)]
    expect(flaggedShare(posts, 'acct').suspected).toBe(true)
  })

  it('handles an unknown handle without flagging', () => {
    expect(flaggedShare([], 'nobody')).toEqual({ flagged: 0, total: 0, suspected: false })
  })
})

describe('truthTag', () => {
  it('returns null for uploads without ground truth', () => {
    const upload = { ...IG_POSTS[0], ground_truth: null } as PostView
    expect(truthTag(upload)).toBeNull()
  })

  it('flags a fake post the detector missed', () => {
    const base = IG_POSTS.find((p) => p.ground_truth === 'fake') as PostView
    const missed: PostView = { ...base, detection: { ...base.detection, flag: 'none', verdict: 'real' } }
    expect(truthTag(missed)).toEqual({ text: 'Missed (fake)', tone: expect.any(String) })
  })
})
