import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { useDetect } from '../hooks/useDetect'
import type { Detection, FeedPost, PostView } from '../types'
import { ExplanationSheet } from '../components/ExplanationSheet'
import { FlagChip, badgeFor } from '../components/FlagChip'
import { UploadSheet } from '../components/UploadSheet'
import { FeedError } from './FeedError'
import { type FeedItem, YOU, flaggedShare, isPending, truthTag } from './feedShared'

type Tab = 'home' | 'explore' | 'dm' | 'profile'
const RING = 'bg-[linear-gradient(45deg,#fdf497,#d6249f)]'
const CHIPS = ['For you', 'Architecture', 'Travel', 'Fashion', 'Art']

const Icon = ({ d, filled = false, label }: { d: string; filled?: boolean; label?: string }) => (
  <svg viewBox="0 0 24 24" width="24" height="24" fill={filled ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" aria-hidden={label ? undefined : true} role={label ? 'img' : undefined} aria-label={label}>
    <path d={d} />
  </svg>
)
const PATHS = {
  home: 'M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z',
  search: 'M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM21 21l-4.3-4.3',
  reels: 'M4 4h16v16H4zM10 9l5 3-5 3z',
  dm: 'M22 3L11 14M22 3l-7 18-4-7-7-4z',
  user: 'M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21a8 8 0 0 1 16 0',
  heart: 'M12 21s-7-4.5-9.5-9A5.5 5.5 0 0 1 12 6a5.5 5.5 0 0 1 9.5 6c-2.5 4.5-9.5 9-9.5 9z',
  comment: 'M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z',
  save: 'M6 3h12v18l-6-4-6 4z',
}

function BottomNav({ tab, onTab, onUpload }: { tab: Tab; onTab: (t: Tab) => void; onUpload: () => void }) {
  const btn = (t: Tab, d: string, label: string) => (
    <button
      type="button"
      onClick={() => onTab(t)}
      aria-label={label}
      aria-current={tab === t ? 'page' : undefined}
      className={`flex flex-1 justify-center py-3 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#0095F6] ${tab === t ? 'text-[#fafafa]' : 'text-[#8e8e8e]'}`}
    >
      <Icon d={d} filled={tab === t && t === 'home'} />
    </button>
  )
  return (
    <nav className="fixed inset-x-0 bottom-0 z-40 mx-auto flex max-w-[430px] items-center border-t border-[#262626] bg-[#121212]">
      {btn('home', PATHS.home, 'Home')}
      {btn('explore', PATHS.search, 'Explore')}
      <Link to="/reels" aria-label="Reels" className="flex flex-1 justify-center py-3 text-[#8e8e8e] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#0095F6]">
        <Icon d={PATHS.reels} />
      </Link>
      <button type="button" onClick={onUpload} aria-label="New post" className="flex flex-1 justify-center py-3 text-[#fafafa] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#0095F6]">
        <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="4" /><path d="M12 8v8M8 12h8" /></svg>
      </button>
      {btn('dm', PATHS.dm, 'Direct messages')}
      {btn('profile', PATHS.user, 'Profile')}
    </nav>
  )
}

function Post({ post, index, reveal, onExplain, onOpenPersona }: { post: PostView; index: number; reveal: boolean; onExplain: (d: Detection) => void; onOpenPersona: (h: string) => void }) {
  const [liked, setLiked] = useState(false)
  const badge = badgeFor(post.detection)
  const tag = reveal ? truthTag(post) : null
  return (
    <article className="border-b border-[#262626] pb-3">
      <header className="flex items-center gap-3 px-3 py-2">
        <div className={`rounded-full p-[2px] ${RING}`}>
          <img src={post.avatar} alt="" className="h-8 w-8 rounded-full border-2 border-[#121212] object-cover" />
        </div>
        <button type="button" onClick={() => onOpenPersona(post.handle)} className="flex-1 truncate text-left text-[14px] font-medium focus:outline-none focus-visible:underline">
          {post.handle}
          <span className="ml-2 text-[13px] font-normal text-[#8e8e8e]">{index + 1}h</span>
        </button>
        <span className="text-[#fafafa]" aria-hidden="true">···</span>
      </header>
      <div className="relative">
        {post.media_type === 'video' ? (
          <video src={post.media} muted playsInline loop controls className="aspect-square w-full bg-black object-cover" />
        ) : (
          <img src={post.media} alt={post.caption} className="aspect-square w-full object-cover" />
        )}
        {tag && <span className={`absolute right-2 top-2 rounded px-2 py-0.5 text-xs font-semibold ${tag.tone}`}>{tag.text}</span>}
      </div>
      <div className="flex items-center gap-4 px-3 pt-2">
        <button type="button" onClick={() => setLiked((v) => !v)} aria-label={liked ? 'Unlike' : 'Like'} aria-pressed={liked} className={liked ? 'text-red-500' : 'text-[#fafafa]'}>
          <Icon d={PATHS.heart} filled={liked} />
        </button>
        <Icon d={PATHS.comment} />
        <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M22 3L11 14M22 3l-7 18-4-7-7-4z" /></svg>
        <span className="ml-auto"><Icon d={PATHS.save} /></span>
      </div>
      <div className="space-y-1 px-3 pt-1 text-[13px]">
        <p className="font-medium">{post.likes + (liked ? 1 : 0)} likes</p>
        <p>
          <span className="font-medium">{post.handle}</span> {post.caption}
        </p>
        {badge && <FlagChip badge={badge} variant="inline" onOpen={() => onExplain(post.detection)} />}
        <p className="text-[#8e8e8e]">View all {post.comments} comments</p>
      </div>
    </article>
  )
}

function Stories({ posts, onOpenPersona }: { posts: PostView[]; onOpenPersona: (h: string) => void }) {
  const seen = new Set<string>()
  const people = posts.filter((p) => (seen.has(p.handle) ? false : (seen.add(p.handle), true))).slice(0, 8)
  return (
    <div className="flex gap-3 overflow-x-auto px-3 py-3" aria-label="Stories">
      {people.map((p) => (
        <button key={p.handle} type="button" onClick={() => onOpenPersona(p.handle)} className="flex w-16 shrink-0 flex-col items-center gap-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#0095F6]">
          <span className={`rounded-full p-[2px] ${RING}`}>
            <img src={p.avatar} alt="" className="h-14 w-14 rounded-full border-2 border-[#121212] object-cover" />
          </span>
          <span className="w-full truncate text-center text-[11px] text-[#fafafa]">{p.handle}</span>
        </button>
      ))}
    </div>
  )
}

function PersonaProfile({ handle, posts, onExplain, onBack }: { handle: string; posts: PostView[]; onExplain: (d: Detection) => void; onBack: () => void }) {
  const mine = posts.filter((p) => p.handle === handle)
  const { flagged, total, suspected } = flaggedShare(posts, handle)
  const [following, setFollowing] = useState(false)
  const avatar = mine[0]?.avatar ?? YOU.avatar
  return (
    <section className="space-y-4 px-4 pt-4">
      <header className="flex items-center gap-4">
        <div className={`rounded-full p-[2px] ${RING}`}>
          <img src={avatar} alt="" className="h-20 w-20 rounded-full border-2 border-[#121212] object-cover" />
        </div>
        <dl className="flex flex-1 justify-around text-center text-[13px]">
          <div><dt className="text-[#8e8e8e]">Posts</dt><dd className="font-medium">{total}</dd></div>
          <div><dt className="text-[#8e8e8e]">Followers</dt><dd className="font-medium">{(mine.reduce((s, p) => s + p.likes, 0) / 10).toFixed(0)}</dd></div>
          <div><dt className="text-[#8e8e8e]">Following</dt><dd className="font-medium">{mine.length * 12}</dd></div>
        </dl>
      </header>
      <div>
        <p className="flex flex-wrap items-center gap-2 text-[14px] font-medium">
          {handle}
          {suspected && (
            <span className="inline-flex items-center gap-1 rounded-full bg-[#0095F6]/15 px-2 py-0.5 text-xs text-[#0095F6]">🛡️ Suspected AI</span>
          )}
        </p>
        <p className="text-[13px] text-[#8e8e8e]">{mine[0]?.caption ?? 'Demo account'}</p>
        <p className="mt-1 text-[12px] text-[#8e8e8e]">
          {flagged} of {total} posts flagged by the detector
        </p>
      </div>
      <div className="flex gap-2">
        <button type="button" onClick={() => setFollowing((v) => !v)} className={`flex-1 rounded-lg py-1.5 text-[14px] font-semibold ${following ? 'bg-[#262626] text-[#fafafa]' : 'bg-[#0095F6] text-white'}`}>
          {following ? 'Following' : 'Follow'}
        </button>
        <button type="button" onClick={onBack} className="rounded-lg bg-[#262626] px-4 py-1.5 text-[14px] font-semibold">
          Back
        </button>
      </div>
      <div className="grid grid-cols-3 gap-[2px]">
        {mine.map((p) => (
          <button key={p.id} type="button" onClick={() => onExplain(p.detection)} className="relative aspect-square focus:outline-none focus-visible:ring-2 focus-visible:ring-[#0095F6]" aria-label={`Post: ${p.caption}`}>
            <img src={p.media} alt="" className="h-full w-full object-cover" />
            {badgeFor(p.detection) && <span className="absolute right-1 top-1 text-[#0095F6] drop-shadow" aria-hidden="true">✨</span>}
          </button>
        ))}
      </div>
    </section>
  )
}

function Explore({ posts }: { posts: PostView[] }) {
  const [chip, setChip] = useState(CHIPS[0])
  return (
    <section className="space-y-3 px-2 pt-3">
      <input type="search" placeholder="Search" aria-label="Search" className="w-full rounded-lg bg-[#262626] px-3 py-2 text-[14px] placeholder:text-[#8e8e8e] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#0095F6]" />
      <div className="flex gap-2 overflow-x-auto" role="group" aria-label="Categories">
        {CHIPS.map((c) => (
          <button key={c} type="button" aria-pressed={chip === c} onClick={() => setChip(c)} className={`shrink-0 rounded-lg px-3 py-1 text-[13px] ${chip === c ? 'bg-[#fafafa] text-[#121212]' : 'bg-[#262626]'}`}>
            {c}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-[2px]">
        {posts.map((p, i) => (
          <div key={p.id} className={`overflow-hidden bg-[#262626] ${i % 7 === 0 ? 'col-span-2 row-span-2' : 'aspect-square'}`}>
            <img src={p.media} alt={p.caption} className="h-full w-full object-cover" />
          </div>
        ))}
      </div>
    </section>
  )
}

function Messages({ posts }: { posts: PostView[] }) {
  const [box, setBox] = useState('Primary')
  const handles = useMemo(() => Array.from(new Set(posts.map((p) => p.handle))), [posts])
  return (
    <section className="px-3 pt-3">
      <div className="flex gap-4 border-b border-[#262626] text-[14px]" role="tablist" aria-label="Inbox">
        {['Primary', 'General', 'Requests'].map((t) => (
          <button key={t} role="tab" aria-selected={box === t} type="button" onClick={() => setBox(t)} className={`pb-2 ${box === t ? 'border-b-2 border-[#fafafa] font-medium' : 'text-[#8e8e8e]'}`}>
            {t}
          </button>
        ))}
      </div>
      <ul className="mt-2 space-y-3">
        {handles.map((h, i) => {
          if (box === 'Requests' ? i % 2 === 0 : box === 'General' ? i % 2 === 1 : false) return null
          return (
            <li key={h} className="flex items-center gap-3 text-[14px]">
              <span className="relative">
                <img src={posts.find((p) => p.handle === h)?.avatar} alt="" className="h-12 w-12 rounded-full object-cover" />
                {i % 3 === 0 && <span className="absolute bottom-0 right-0 h-3 w-3 rounded-full border-2 border-[#121212] bg-[#0095F6]" aria-label="Online" />}
              </span>
              <span className="flex-1">
                <span className={i % 4 === 1 ? 'font-semibold' : ''}>{h}</span>
                <span className="block text-[13px] text-[#8e8e8e]">Sent a message · {i + 1}h</span>
              </span>
              {i % 4 === 1 && <span className="h-2 w-2 rounded-full bg-[#0095F6]" aria-label="Unread" />}
            </li>
          )
        })}
      </ul>
    </section>
  )
}

export function Instagram() {
  const [posts, setPosts] = useState<PostView[]>([])
  const [items, setItems] = useState<FeedItem[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [reveal, setReveal] = useState(false)
  const [explain, setExplain] = useState<Detection | null>(null)
  const [uploading, setUploading] = useState(false)
  const [tab, setTab] = useState<Tab>('home')
  const [persona, setPersona] = useState<string | null>(null)
  const { run, error: scanError } = useDetect()

  useEffect(() => {
    api<{ posts: FeedPost[] }>('/api/feed/instagram')
      .then((r) => setPosts(r.posts))
      .catch((e: Error) => setLoadError(e.message))
  }, [])

  const onUpload = useCallback(
    async (file: File, caption: string) => {
      setUploading(false)
      const id = crypto.randomUUID()
      const url = URL.createObjectURL(file)
      const mediaType = file.type.startsWith('video/') ? 'video' : 'image'
      setItems([{ id, pending: true, media: url, media_type: mediaType, caption }])
      const detection = await run(file)
      setItems([])
      if (detection) {
        setPosts((p) => [{ id, handle: YOU.handle, avatar: YOU.avatar, caption, media: url, media_type: mediaType, likes: 0, comments: 0, ground_truth: null, source: YOU.source, detection }, ...p])
        setTab('home')
      }
    },
    [run],
  )

  const openPersona = (h: string) => {
    setPersona(h)
    setTab('profile')
  }

  return (
    <div className="mx-auto min-h-dvh max-w-[430px] bg-[#121212] pb-20 text-[#fafafa]">
      <header className="sticky top-0 z-30 flex items-center gap-3 bg-[#121212] px-4 py-3">
        <span className="flex-1 font-serif text-[22px] italic tracking-tight">Spectra Guard</span>
        <label className="flex cursor-pointer items-center gap-1.5 text-[11px] text-[#8e8e8e]">
          <input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} className="h-3.5 w-3.5 accent-[#0095F6]" />
          Ground truth
        </label>
        <Link to="/" className="text-[12px] text-[#8e8e8e] underline focus-visible:ring-2 focus-visible:ring-[#0095F6]">Playground</Link>
      </header>

      <main>
        {loadError && <FeedError message={loadError} />}
        {scanError && <p role="alert" className="px-4 pt-3 text-[13px] text-red-400">{scanError}</p>}

        {tab === 'home' && (
          <>
            <Stories posts={posts} onOpenPersona={openPersona} />
            {items.map((it) => (
              <article key={it.id} className="border-b border-[#262626] pb-3" aria-busy="true">
                <header className="flex items-center gap-3 px-3 py-2">
                  <img src={YOU.avatar} alt="" className="h-8 w-8 rounded-full object-cover" />
                  <span className="text-[14px] font-medium">{YOU.handle}</span>
                </header>
                <div className="relative">
                  {isPending(it) && it.media_type === 'video' ? (
                    <video src={it.media} muted playsInline className="aspect-square w-full object-cover" />
                  ) : (
                    <img src={it.media} alt="Scanning upload" className="aspect-square w-full object-cover" />
                  )}
                  <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-[14px] font-semibold" role="status">Scanning...</div>
                </div>
              </article>
            ))}
            {posts.map((p, i) => (
              <Post key={p.id} post={p} index={i} reveal={reveal} onExplain={setExplain} onOpenPersona={openPersona} />
            ))}
          </>
        )}
        {tab === 'explore' && <Explore posts={posts} />}
        {tab === 'dm' && <Messages posts={posts} />}
        {tab === 'profile' && (
          persona ? (
            <PersonaProfile handle={persona} posts={posts} onExplain={setExplain} onBack={() => setPersona(null)} />
          ) : (
            <PersonaProfile handle={YOU.handle} posts={posts} onExplain={setExplain} onBack={() => setTab('home')} />
          )
        )}
      </main>

      <BottomNav tab={tab} onTab={(t) => { setPersona(null); setTab(t) }} onUpload={() => setUploading(true)} />
      {uploading && <UploadSheet onClose={() => setUploading(false)} onSubmit={onUpload} dark />}
      {explain && <ExplanationSheet detection={explain} onClose={() => setExplain(null)} />}
    </div>
  )
}
