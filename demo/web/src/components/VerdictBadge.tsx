import type { Verdict } from '../types'
import { VERDICT_LABEL } from '../types'

const TONE: Record<Verdict, string> = {
  fake: 'bg-red-500/15 text-red-400 ring-red-500/40',
  review: 'bg-amber-500/15 text-amber-300 ring-amber-500/40',
  real: 'bg-emerald-500/15 text-emerald-400 ring-emerald-500/40',
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return (
    <span className={`inline-flex rounded-md px-2 py-0.5 text-xs font-semibold ring-1 ${TONE[verdict]}`}>
      {VERDICT_LABEL[verdict]}
    </span>
  )
}
