import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import type { Detection, FeedPost, PostView } from '../types'
import { ExplanationSheet } from '../components/ExplanationSheet'
import { FlagChip, badgeFor } from '../components/FlagChip'
import { FeedError } from './FeedError'
import { truthTag } from './feedShared'

// Plays only while at least 60% visible, so muted autoplay does not run for every reel.
function AutoVideo({ src }: { src: string }) {
  const ref = useRef<HTMLVideoElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && entry.intersectionRatio >= 0.6) void el.play().catch(() => undefined)
        else el.pause()
      },
      { threshold: [0, 0.6] },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return <video ref={ref} src={src} autoPlay muted loop playsInline className="absolute inset-0 h-full w-full bg-black object-cover" />
}

// PRD 3.2: full-bleed 9:16 reel, creator + audio attribution, shield overlay, tap to inspect.
function Reel({ post, reveal, onExplain }: { post: PostView; reveal: boolean; onExplain: (d: Detection) => void }) {
  const badge = badgeFor(post.detection)
  const tag = reveal ? truthTag(post) : null
  return (
    <section className="relative h-[100dvh] w-full snap-start overflow-hidden bg-black text-white" aria-label={`Reel by ${post.handle}`}>
      {post.media_type === 'video' ? (
        <AutoVideo src={post.media} />
      ) : (
        <img src={post.media} alt={post.caption} className="absolute inset-0 h-full w-full object-cover" />
      )}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/20" aria-hidden="true" />

      <div className="absolute bottom-24 left-4 right-20 space-y-2 text-[14px] drop-shadow-[0_1px_3px_rgba(0,0,0,0.9)]">
        {badge && (
          <div>
            <FlagChip badge={badge} variant="overlay" onOpen={() => onExplain(post.detection)} />
          </div>
        )}
        {tag && <span className={`inline-block rounded px-2 py-0.5 text-xs font-semibold ${tag.tone}`}>{tag.text}</span>}
        <p className="font-medium">{post.handle}</p>
        <p>{post.caption}</p>
        <p className="text-[13px] opacity-90">♫ Original audio · {post.handle}</p>
      </div>
    </section>
  )
}

export function Reels() {
  const [posts, setPosts] = useState<PostView[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [reveal, setReveal] = useState(false)
  const [explain, setExplain] = useState<Detection | null>(null)

  useEffect(() => {
    api<{ posts: FeedPost[] }>('/api/feed/tiktok')
      .then((r) => setPosts(r.posts))
      .catch((e: Error) => setLoadError(e.message))
  }, [])

  return (
    <div className="relative h-[100dvh] snap-y snap-mandatory overflow-y-auto bg-black">
      <div className="pointer-events-none fixed inset-x-0 top-0 z-20 flex items-start justify-between p-4">
        <Link to="/instagram" className="pointer-events-auto rounded-full bg-black/40 px-3 py-1 text-sm text-white focus-visible:ring-2 focus-visible:ring-[#0095F6]">
          Spectra Guard
        </Link>
        <span className="text-[15px] font-semibold text-white">Reels</span>
        <label className="pointer-events-auto flex items-center gap-1 rounded-full bg-black/40 px-3 py-1 text-xs text-white">
          <input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} className="accent-[#0095F6]" />
          Ground truth
        </label>
      </div>
      {loadError && <FeedError message={loadError} />}
      {posts.map((p) => (
        <Reel key={p.id} post={p} reveal={reveal} onExplain={setExplain} />
      ))}
      {explain && <ExplanationSheet detection={explain} onClose={() => setExplain(null)} />}
    </div>
  )
}
