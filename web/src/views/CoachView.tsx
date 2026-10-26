import { Title } from '../components/ui'

const ASKS = [
  'Two games this week, Monday and Wednesday. Plan it.',
  'What does a winger do most in 5-a-side?',
  'My hips feel tight after Sunday. Adjust tomorrow.',
  'Build me a 45-minute weak-foot session for the wall.',
]

// Placeholder until the agent backend exists. Shows what it will do; nothing pretends to work.
export function CoachView({ wide = false }: { wide?: boolean }) {
  return (
    <main className={wide ? 'max-w-3xl px-10 pt-10 pb-12' : 'mx-auto max-w-xl px-4 pt-[max(20px,env(safe-area-inset-top))] pb-32'}>
      <Title>Coach</Title>
      <p className="m-0 mt-3 max-w-[60ch] text-muted">
        Your training agent. Talk through your week, games and how your body feels. It researches drills for your positions,
        reads your check-ins and reshapes the plan.
      </p>

      <section className="mt-8 rounded-[24px] bg-surface p-5">
        <h2 className="m-0 font-display text-base uppercase">Things you'll ask</h2>
        <ul className="m-0 mt-4 flex list-none flex-col gap-2 p-0">
          {ASKS.map((a) => (
            <li key={a} className="rounded-2xl bg-surface-2 px-4 py-3 text-[15px]">{a}</li>
          ))}
        </ul>
        <div className="mt-5 flex items-center gap-3 rounded-full bg-bg px-5 py-3 text-muted" aria-disabled="true">
          <span className="flex-1">Message your coach…</span>
          <span className="rounded-full bg-surface-2 px-3 py-1 font-display text-[11px] uppercase">Next build</span>
        </div>
      </section>

      <p className="m-0 mt-4 text-sm text-muted">Coming with the backend: Gemini with Groq as fallback, your data kept on your own server.</p>
    </main>
  )
}
