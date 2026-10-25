import { ApiError } from '../api'
import { IG_POSTS, MODELS, TT_POSTS, buildDetection } from './fixtures'

const SESSION_KEY = 'dfd_mock_session'
const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const VIDEO_TYPES = ['video/mp4', 'video/quicktime', 'video/webm']
const DETECT_DELAY_MS = 800

const hasSession = () => sessionStorage.getItem(SESSION_KEY) === '1'
function fail(status: number, error: string): never {
  throw new ApiError(status, error)
}

// Random but schema-valid detection; the fusion rule matches the contract.
async function mockDetect(form: FormData): Promise<unknown> {
  const file = form.get('file')
  if (!(file instanceof File)) fail(400, 'missing file')
  const isImage = IMAGE_TYPES.includes(file.type)
  const isVideo = VIDEO_TYPES.includes(file.type)
  if (!isImage && !isVideo) fail(415, 'unsupported media type')
  if (isImage && file.size > 15 * 1024 * 1024) fail(413, 'image larger than 15 MB')
  if (isVideo && file.size > 50 * 1024 * 1024) fail(413, 'video larger than 50 MB')
  await new Promise((r) => setTimeout(r, DETECT_DELAY_MS))
  const id = crypto.randomUUID()
  const rnd = () => Math.random()
  if (isImage) {
    const bmp = await createImageBitmap(file)
    const w = bmp.width
    const h = bmp.height
    bmp.close()
    const hasFace = rnd() < 0.5
    const boxW = Math.round(w * 0.3)
    const boxH = Math.round(h * 0.4)
    const box: [number, number, number, number] = [Math.round(w * 0.35), Math.round(h * 0.2), boxW, boxH]
    return buildDetection({
      id,
      mediaType: 'image',
      general: rnd(),
      face: hasFace ? { p: rnd(), box, score: 0.5 + rnd() / 2 } : undefined,
      latency: 800 + Math.round(rnd() * 200),
    })
  }
  const frames = Array.from({ length: 8 }, () => rnd())
  return buildDetection({ id, mediaType: 'video', general: Math.max(...frames), frames, latency: 1200 })
}

export async function mockApi<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? 'GET').toUpperCase()
  const url = new URL(path, 'http://mock.local')
  const route = url.pathname

  if (route === '/api/login' && method === 'POST') {
    const body = JSON.parse(String(init?.body ?? '{}')) as { passcode?: string }
    if (body.passcode !== 'demo') fail(401, 'wrong passcode')
    sessionStorage.setItem(SESSION_KEY, '1')
    return { ok: true } as T
  }
  if (route === '/api/session') {
    if (!hasSession()) fail(401, 'not signed in')
    return { ok: true } as T
  }
  if (!hasSession()) fail(401, 'not signed in')

  if (route === '/api/feed/instagram') return { posts: IG_POSTS } as T
  if (route === '/api/feed/tiktok') return { posts: TT_POSTS } as T
  if (route === '/api/models') return MODELS as T
  if (route === '/api/detect' && method === 'POST') return (await mockDetect(init?.body as FormData)) as T
  fail(404, `no mock route for ${method} ${route}`)
  throw new Error('unreachable')
}
