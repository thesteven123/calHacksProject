import type { Detection, FeedPost, Flag, Specialist, Verdict, ModelsResponse } from '../types'

// Thresholds are demo-only; the real service uses calibrated per-specialist thresholds.
const FAKE_AT = 0.9
const REVIEW_AT = 0.5

const LABEL: Record<Verdict, string> = {
  fake: 'Likely AI-generated or manipulated',
  review: 'Unverified - needs human review',
  real: 'No manipulation detected',
}
const FLAG: Record<Verdict, Flag> = { fake: 'ai_generated', review: 'needs_review', real: 'none' }
const DISCLAIMER = 'Research demo. Not for enforcement decisions; uncertain cases go to human review.'

const verdictOf = (p: number): Verdict => (p >= FAKE_AT ? 'fake' : p >= REVIEW_AT ? 'review' : 'real')

interface Spec {
  id: string
  mediaType: 'image' | 'video'
  general: number
  face?: { p: number; box: [number, number, number, number]; score: number }
  frames?: number[]
  latency?: number
}

// Builds a schema-valid Detection with the contract's fusion rule.
export function buildDetection(s: Spec): Detection {
  const specialists: Specialist[] = [
    {
      name: 'general',
      title: 'General AI-image detector',
      applies_to: s.mediaType === 'video' ? 'keyframes' : 'whole image',
      p_fake: s.general,
      verdict: verdictOf(s.general),
      calibration_status: 'matched',
    },
  ]
  const faces: Detection['faces'] = []
  if (s.face) {
    specialists.push({
      name: 'face',
      title: 'AI-generated face detector',
      applies_to: 'face crop',
      p_fake: s.face.p,
      verdict: verdictOf(s.face.p),
      calibration_status: 'matched',
      box: s.face.box,
    })
    faces.push({ box: s.face.box, score: s.face.score })
  }
  const verdicts = specialists.map((x) => x.verdict)
  const verdict: Verdict = verdicts.includes('fake')
    ? 'fake'
    : verdicts.every((v) => v === 'real')
      ? 'real'
      : 'review'
  const deciding = specialists.filter((x) => x.verdict === verdict)
  const confidence =
    verdict === 'real'
      ? 1 - Math.max(...specialists.map((x) => x.p_fake))
      : Math.max(...deciding.map((x) => x.p_fake))
  const reasons = specialists.map((x) =>
    x.name === 'face'
      ? `The face crop scores ${x.p_fake.toFixed(2)} on the AI-face detector (threshold ${FAKE_AT}).`
      : `The general detector scores ${x.p_fake.toFixed(2)} (threshold ${FAKE_AT}).`,
  )
  return {
    id: s.id,
    media_type: s.mediaType,
    verdict,
    flag: FLAG[verdict],
    label: LABEL[verdict],
    confidence: Number(confidence.toFixed(3)),
    specialists,
    faces,
    video:
      s.mediaType === 'video'
        ? {
            method: 'keyframe_heuristic',
            calibrated: false,
            n_keyframes: s.frames?.length ?? 0,
            frames: (s.frames ?? []).map((p, i) => ({
              t: Number((i * 0.75).toFixed(2)),
              verdict: verdictOf(p),
              p_fake: p,
            })),
          }
        : null,
    reasons,
    latency_ms: s.latency ?? 390,
    model_versions: { general: 'tiny_genimage_all', face: 'face_t2i_b3' },
    disclaimer: DISCLAIMER,
  }
}

const IG_SOURCE = 'Synthetic placeholder (mock mode)'

export const IG_POSTS: FeedPost[] = [
  { id: 'ig_001', handle: 'sunny.trails', avatar: '/media/avatars/a01.jpg', caption: 'Morning hike, the light was perfect.', media: '/media/instagram/ig_001.jpg', media_type: 'image', likes: 1234, comments: 56, ground_truth: 'real', source: IG_SOURCE,
    detection: buildDetection({ id: 'ig_001', mediaType: 'image', general: 0.08, face: { p: 0.12, box: [180, 160, 240, 260], score: 0.98 } }) },
  { id: 'ig_002', handle: 'pixel.and.pine', avatar: '/media/avatars/a02.jpg', caption: 'Generated this in a dream. Thoughts?', media: '/media/instagram/ig_002.jpg', media_type: 'image', likes: 802, comments: 41, ground_truth: 'fake', source: IG_SOURCE,
    detection: buildDetection({ id: 'ig_002', mediaType: 'image', general: 0.97, face: { p: 0.95, box: [200, 150, 230, 250], score: 0.99 } }) },
  { id: 'ig_003', handle: 'cafe.notes', avatar: '/media/avatars/a03.jpg', caption: 'Coffee and a window seat.', media: '/media/instagram/ig_003.jpg', media_type: 'image', likes: 415, comments: 9, ground_truth: 'real', source: IG_SOURCE,
    detection: buildDetection({ id: 'ig_003', mediaType: 'image', general: 0.31 }) },
  { id: 'ig_004', handle: 'river.lens', avatar: '/media/avatars/a04.jpg', caption: 'Lens test, shot on a phone.', media: '/media/instagram/ig_004.jpg', media_type: 'image', likes: 2210, comments: 130, ground_truth: 'fake', source: IG_SOURCE,
    detection: buildDetection({ id: 'ig_004', mediaType: 'image', general: 0.62 }) },
  { id: 'ig_005', handle: 'studio.oak', avatar: '/media/avatars/a05.jpg', caption: 'Studio day. New series soon.', media: '/media/instagram/ig_005.jpg', media_type: 'image', likes: 97, comments: 3, ground_truth: 'fake', source: IG_SOURCE,
    detection: buildDetection({ id: 'ig_005', mediaType: 'image', general: 0.04 }) },
  { id: 'ig_006', handle: 'the.kitchen.log', avatar: '/media/avatars/a06.jpg', caption: 'Sunday bread, third try.', media: '/media/instagram/ig_006.jpg', media_type: 'image', likes: 640, comments: 22, ground_truth: 'real', source: IG_SOURCE,
    detection: buildDetection({ id: 'ig_006', mediaType: 'image', general: 0.05 }) },
]

export const TT_POSTS: FeedPost[] = [
  { id: 'tt_001', handle: 'dance.loop', avatar: '/media/avatars/a02.jpg', caption: 'Practice clip #14 #dance', media: '/media/tiktok/tt_001.mp4', media_type: 'video', likes: 18300, comments: 412, ground_truth: 'real', source: IG_SOURCE,
    detection: buildDetection({ id: 'tt_001', mediaType: 'video', general: 0.1, frames: [0.08, 0.12, 0.1, 0.09, 0.11, 0.07, 0.1, 0.13] }) },
  { id: 'tt_002', handle: 'virtual.host', avatar: '/media/avatars/a04.jpg', caption: 'Hello from the future. Not a real person.', media: '/media/tiktok/tt_002.mp4', media_type: 'video', likes: 9120, comments: 233, ground_truth: 'fake', source: IG_SOURCE,
    detection: buildDetection({ id: 'tt_002', mediaType: 'video', general: 0.96, frames: [0.93, 0.96, 0.97, 0.91, 0.95, 0.94, 0.9, 0.96] }) },
  { id: 'tt_003', handle: 'street.snacks', avatar: '/media/avatars/a06.jpg', caption: 'Tasting the new noodle spot', media: '/media/tiktok/tt_003.mp4', media_type: 'video', likes: 3340, comments: 77, ground_truth: 'fake', source: IG_SOURCE,
    detection: buildDetection({ id: 'tt_003', mediaType: 'video', general: 0.66, frames: [0.6, 0.66, 0.7, 0.55, 0.62, 0.68, 0.59, 0.64] }) },
  { id: 'tt_004', handle: 'night.drive', avatar: '/media/avatars/a05.jpg', caption: 'City lights at 2am', media: '/media/tiktok/tt_004.jpg', media_type: 'image', likes: 5711, comments: 150, ground_truth: 'real', source: IG_SOURCE,
    detection: buildDetection({ id: 'tt_004', mediaType: 'image', general: 0.02 }) },
]

export const MODELS: ModelsResponse = {
  specialists: [
    {
      name: 'general',
      title: 'General AI-image detector',
      checkpoint: 'runs/tiny_genimage_all/best.pt',
      trained_on: 'Tiny-GenImage (research license, CC-BY-NC-SA)',
      metrics: { note: 'mock-mode placeholder numbers' },
      limitations: ['Trained on a limited set of generators.', 'Compression can hide artefacts.'],
    },
    {
      name: 'face',
      title: 'AI-generated face detector',
      checkpoint: 'runs/face_t2i_b3/best.pt',
      trained_on: 'Face crops from text-to-image generators',
      metrics: { note: 'mock-mode placeholder numbers' },
      limitations: ['Only scores detected faces.', 'Heavy filters can cause false positives.'],
    },
  ],
  fusion: 'fake if any specialist says fake; real only if all say real; else review',
  notes: ['Mock mode: all responses are fixtures or random demo output.'],
  api_url: undefined,
}
