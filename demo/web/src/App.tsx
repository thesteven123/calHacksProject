import { useEffect, useState } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { api } from './api'
import { Login } from './pages/Login'
import { Playground } from './pages/Playground'
import { ApiPage } from './pages/ApiPage'
import { Instagram } from './pages/Instagram'
import { Reels } from './pages/Reels'

type Gate = 'checking' | 'out' | 'in'

export default function App() {
  const [gate, setGate] = useState<Gate>('checking')

  useEffect(() => {
    api('/api/session')
      .then(() => setGate('in'))
      .catch(() => setGate('out'))
  }, [])

  if (gate === 'checking') return <div className="min-h-dvh bg-neutral-950" aria-busy="true" />
  if (gate === 'out') return <Login onAuthed={() => setGate('in')} />

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Playground />} />
        <Route path="/developers" element={<ApiPage />} />
        <Route path="/instagram" element={<Instagram />} />
        <Route path="/reels" element={<Reels />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
