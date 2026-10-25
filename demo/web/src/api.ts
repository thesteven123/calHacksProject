import { mockApi } from './mock/mockApi'

const MOCK = import.meta.env.VITE_MOCK === '1'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

// Same-origin /api/* in real mode; fixtures in VITE_MOCK=1 mode.
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  if (MOCK) return mockApi<T>(path, init)
  const res = await fetch(path, { credentials: 'same-origin', ...init })
  if (!res.ok) {
    let message = res.statusText
    try {
      const body = (await res.json()) as { error?: string }
      if (body.error) message = body.error
    } catch {
      // non-JSON error body: keep statusText
    }
    throw new ApiError(res.status, message)
  }
  return (await res.json()) as T
}

export const isMock = MOCK
