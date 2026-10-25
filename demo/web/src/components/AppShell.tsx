import { NavLink } from 'react-router-dom'
import type { ReactNode } from 'react'

const LINKS = [
  { to: '/', label: 'Playground' },
  { to: '/instagram', label: 'Spectra Guard' },
  { to: '/reels', label: 'Reels' },
  { to: '/developers', label: 'API' },
]

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `rounded-lg px-3 py-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
    isActive ? 'bg-neutral-800 text-white' : 'text-neutral-400 hover:bg-neutral-900 hover:text-neutral-200'
  }`

// Playground-style layout: sidebar on desktop, compact link bar on small screens.
export function AppShell({ children, actions }: { children: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex h-dvh flex-col bg-neutral-950 text-neutral-100 md:flex-row">
      <nav className="flex gap-1 overflow-x-auto border-b border-neutral-800 p-2 md:hidden" aria-label="Main">
        {LINKS.map((l) => (
          <NavLink key={l.to} to={l.to} end={l.to === '/'} className={linkClass}>
            {l.label}
          </NavLink>
        ))}
      </nav>
      <aside className="hidden w-60 shrink-0 flex-col gap-1 border-r border-neutral-800 bg-neutral-900/40 p-3 md:flex" aria-label="Sidebar">
        <div className="px-3 py-2 text-base font-semibold">DFD Detector</div>
        {actions}
        {LINKS.map((l) => (
          <NavLink key={l.to} to={l.to} end={l.to === '/'} className={linkClass}>
            {l.label}
          </NavLink>
        ))}
      </aside>
      <div className="flex min-h-0 flex-1 flex-col">{children}</div>
    </div>
  )
}
