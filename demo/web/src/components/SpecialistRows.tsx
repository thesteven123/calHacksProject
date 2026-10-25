import type { Specialist, Verdict } from '../types'

const BAR: Record<Verdict, string> = {
  fake: 'bg-red-500',
  review: 'bg-amber-400',
  real: 'bg-emerald-500',
}

// One row per specialist: title, what it scores, and a p_fake bar in the verdict colour.
export function SpecialistRows({ specialists }: { specialists: Specialist[] }) {
  return (
    <ul className="space-y-3">
      {specialists.map((s) => (
        <li key={s.name}>
          <div className="flex items-baseline justify-between gap-2 text-sm">
            <span className="font-medium text-neutral-100">{s.title}</span>
            <span className="tabular-nums text-neutral-300">{Math.round(s.p_fake * 100)}%</span>
          </div>
          <div className="text-xs text-neutral-400">Applies to: {s.applies_to}</div>
          <div
            className="mt-1 h-2 w-full overflow-hidden rounded-full bg-neutral-700"
            role="meter"
            aria-label={`${s.title} p_fake`}
            aria-valuemin={0}
            aria-valuemax={1}
            aria-valuenow={s.p_fake}
          >
            <div className={`h-full rounded-full ${BAR[s.verdict]}`} style={{ width: `${Math.round(s.p_fake * 100)}%` }} />
          </div>
        </li>
      ))}
    </ul>
  )
}
