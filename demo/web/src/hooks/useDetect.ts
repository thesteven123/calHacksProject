import { useCallback, useState } from 'react'
import { api } from '../api'
import type { Detection } from '../types'

// Uploads one file to POST /api/detect and tracks busy/error state.
export function useDetect() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = useCallback(async (file: File): Promise<Detection | null> => {
    setBusy(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      return await api<Detection>('/api/detect', { method: 'POST', body: form })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Scan failed')
      return null
    } finally {
      setBusy(false)
    }
  }, [])

  return { run, busy, error }
}
