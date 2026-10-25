import type { PostView } from '../types'

// Optimistic item shown while /api/detect runs; replaced by the real post on success.
export interface PendingItem {
  id: string
  pending: true
  media: string
  media_type: 'image' | 'video'
  caption: string
}

export type FeedItem = PostView | PendingItem

export const isPending = (x: FeedItem): x is PendingItem => 'pending' in x

export function truthTag(post: PostView): { text: string; tone: string } | null {
  if (post.ground_truth === null) return null
  if (post.ground_truth === 'fake' && post.detection.flag === 'none') {
    return { text: 'Missed (fake)', tone: 'bg-fuchsia-600 text-white' }
  }
  return post.ground_truth === 'fake'
    ? { text: 'Truth: fake', tone: 'bg-neutral-900/80 text-red-300' }
    : { text: 'Truth: real', tone: 'bg-neutral-900/80 text-emerald-300' }
}

export const YOU = { handle: 'you', avatar: '/media/avatars/a06.jpg', source: 'Your upload' }

// Account-level heuristic from PRD 5: >60% flagged posts -> "Suspected AI" pill on the profile.
export function flaggedShare(posts: PostView[], handle: string): { flagged: number; total: number; suspected: boolean } {
  const mine = posts.filter((p) => p.handle === handle)
  const flagged = mine.filter((p) => p.detection.flag !== 'none').length
  return { flagged, total: mine.length, suspected: mine.length > 0 && flagged / mine.length > 0.6 }
}
