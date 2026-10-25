// Mirrors runs/helper/API_CONTRACT.md. Keep in sync with the Modal service.
export type Verdict = 'fake' | 'review' | 'real'
export type Flag = 'ai_generated' | 'needs_review' | 'none'

export interface Specialist {
  name: string
  title: string
  applies_to: string
  p_fake: number
  verdict: Verdict
  calibration_status: string
  box?: [number, number, number, number]
}

export interface VideoFrame {
  t: number
  verdict: Verdict
  p_fake: number
}

export interface Detection {
  id: string
  media_type: 'image' | 'video'
  verdict: Verdict
  flag: Flag
  label: string
  confidence: number
  specialists: Specialist[]
  faces: { box: [number, number, number, number]; score: number }[]
  video: null | { method: string; calibrated: boolean; n_keyframes: number; frames: VideoFrame[] }
  reasons: string[]
  latency_ms: number
  model_versions: Record<string, string>
  disclaimer: string
}

export type GroundTruth = 'real' | 'fake'

export interface FeedPost {
  id: string
  handle: string
  avatar: string
  caption: string
  media: string
  media_type: 'image' | 'video'
  likes: number
  comments: number
  ground_truth: GroundTruth
  source: string
  detection: Detection
}

export interface ModelInfo {
  name: string
  title: string
  checkpoint: string
  trained_on: string
  metrics: Record<string, unknown>
  limitations: string[]
}

export interface ModelsResponse {
  specialists: ModelInfo[]
  fusion: string
  notes: string[]
  api_url?: string
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  fake: 'AI-generated',
  review: 'Needs review',
  real: 'Real',
}

// Feed post as rendered; uploads from this session have no ground truth.
export type PostView = Omit<FeedPost, 'ground_truth'> & { ground_truth: GroundTruth | null }
