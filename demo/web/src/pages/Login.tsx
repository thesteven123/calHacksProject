import { useState } from 'react'
import type { FormEvent } from 'react'
import { api, ApiError } from '../api'

// Passcode gate shown when GET /api/session returns 401.
export function Login({ onAuthed }: { onAuthed: () => void }) {
  const [passcode, setPasscode] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await api('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ passcode }) })
      onAuthed()
    } catch (err) {
      setError(err instanceof ApiError && err.status === 401 ? 'Wrong passcode.' : 'Could not sign in. Try again.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="flex min-h-dvh items-center justify-center bg-neutral-950 p-6 text-neutral-100">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-2xl bg-neutral-900 p-6 shadow-xl ring-1 ring-neutral-800">
        <div>
          <h1 className="text-xl font-semibold">DFD Detector</h1>
          <p className="mt-1 text-sm text-neutral-400">Enter the demo passcode to continue.</p>
        </div>
        <label className="block text-sm">
          <span className="mb-1 block text-neutral-300">Passcode</span>
          <input
            type="password"
            autoFocus
            autoComplete="current-password"
            value={passcode}
            onChange={(e) => setPasscode(e.target.value)}
            className="w-full rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          />
        </label>
        {error && (
          <p role="alert" className="text-sm text-red-400">
            {error}
          </p>
        )}
        <button
          type="submit"
          disabled={busy || passcode.length === 0}
          className="w-full rounded-lg bg-sky-600 px-4 py-2 font-semibold text-white disabled:opacity-40"
        >
          {busy ? 'Checking...' : 'Continue'}
        </button>
      </form>
    </main>
  )
}
