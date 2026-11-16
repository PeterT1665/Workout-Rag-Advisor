import { useCallback, useEffect, useState } from 'react'
import { Button, Title } from '../components/ui'
import { FlameIcon } from '../components/icons'
import { Sheet } from '../components/Sheet'
import { CropSheet } from '../components/CropSheet'
import { PlayerCard, overall } from '../components/PlayerCard'
import { cardStatsFor } from '../lib/cardStats'
import { useProfile, type Accent, type Foot, type Position, type Profile } from '../lib/profile'
import { syncEnabled } from '../lib/sync'
import {
  USERNAME, acceptFriend, activity, askFriend, avatarUrls, claimUsername, cleanUsername, findPlayer, friendRequests,
  leaderboard, myTeamProfile, removeFriend, setAvatar, type FoundPlayer, type FriendRequest, type TeamProfile,
} from '../lib/team'
import type { Streak } from '../lib/streak'

// Team: add friends by username + shirt number, accept requests, and a friends leaderboard (streak, then rating).
// Signed-in players only; guests see themselves and an invitation to make an account.
export function TeamView({ streak, onSignUp }: { streak: Streak; onSignUp?: () => void }) {
  return syncEnabled() ? <TeamSignedIn streak={streak} /> : <TeamGuest streak={streak} onSignUp={onSignUp} />
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main className="mx-auto max-w-xl px-4 pt-[max(20px,var(--sat))] pb-32">
      <Title>Team</Title>
      {children}
    </main>
  )
}

function TeamGuest({ streak, onSignUp }: { streak: Streak; onSignUp?: () => void }) {
  const me = useProfile()
  const stats = cardStatsFor(me)
  return (
    <Shell>
      <p className="m-0 mt-2 text-muted">Add your friends, see who's training, and compete on streaks and ratings.</p>
      <section className="mt-6 rounded-[24px] bg-surface p-4" aria-label="Leaderboard">
        <h2 className="m-0 mb-3 font-display text-base uppercase">Leaderboard</h2>
        <Row rank={1} name={me.name || 'You'} number={me.number} sub="You" streak={streak.days} lit={streak.today} rating={stats ? overall(stats) : null} />
      </section>
      <section className="mt-4 rounded-[24px] bg-surface p-4">
        <h2 className="m-0 font-display text-base uppercase">Play with friends</h2>
        <p className="m-0 mt-1 text-sm text-muted">Team needs an account: pick a username, then find friends by username and shirt number.</p>
        {onSignUp && <Button block className="mt-3" onClick={onSignUp}>Create an account</Button>}
      </section>
    </Shell>
  )
}

function TeamSignedIn({ streak }: { streak: Streak }) {
  const me = useProfile()
  const [mine, setMine] = useState<TeamProfile | null | undefined>(undefined) // undefined: loading
  const [board, setBoard] = useState<TeamProfile[]>([])
  const [requests, setRequests] = useState<FriendRequest[]>([])
  const [urls, setUrls] = useState<Record<string, string>>({})
  const [open, setOpen] = useState<TeamProfile | null>(null)
  const [renaming, setRenaming] = useState(false)
  const [offline, setOffline] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const p = await myTeamProfile()
      setMine(p)
      if (!p) return
      const [b, r] = await Promise.all([leaderboard(), friendRequests()])
      setBoard(b); setRequests(r); setOffline(false)
      setUrls(await avatarUrls([...b.map((x) => x.avatar_path), ...r.map((x) => x.avatar_path)]))
    } catch { setOffline(true); setMine((m) => m ?? null) }
  }, [])
  useEffect(() => {
    void refresh() // eslint-disable-line react-hooks/set-state-in-effect
    const t = setInterval(() => { if (document.visibilityState === 'visible') void refresh() }, 60000)
    return () => clearInterval(t)
  }, [refresh])

  if (mine === undefined) return <Shell><p className="mt-4 text-muted" aria-busy="true">Loading your team…</p></Shell>
  if (offline && !mine) return <Shell><p className="mt-4 text-muted">You're offline. Team needs a connection.</p></Shell>
  if (!mine || renaming) {
    return (
      <Shell>
        <UsernameForm current={mine?.username} profile={me} onDone={() => { setRenaming(false); void refresh() }} onCancel={mine ? () => setRenaming(false) : undefined} />
      </Shell>
    )
  }

  const incoming = requests.filter((r) => r.incoming)
  const outgoing = requests.filter((r) => !r.incoming)
  return (
    <Shell>
      <div className="mt-4 flex items-center gap-3">
        <AvatarPicker url={mine.avatar_path ? urls[mine.avatar_path] : undefined} name={mine.display_name || mine.username} onChanged={refresh} />
        <div className="min-w-0 flex-1">
          <p className="m-0 truncate font-semibold">@{mine.username} <span className="text-muted">#{mine.number}</span></p>
          <p className="m-0 text-sm text-muted">Friends find you with your username and shirt number.</p>
        </div>
        <button onClick={() => setRenaming(true)} className="min-h-10 shrink-0 rounded-full bg-surface-2 px-3 text-xs font-semibold">Edit</button>
      </div>

      {incoming.length > 0 && (
        <section className="rise mt-6 rounded-[24px] border border-red/50 bg-surface p-4" aria-label="Friend requests">
          <h2 className="m-0 mb-3 font-display text-base uppercase">Friend requests</h2>
          <ul className="m-0 flex list-none flex-col gap-2 p-0">
            {incoming.map((r) => (
              <li key={r.user_id} className="flex items-center gap-3">
                <Avatar url={r.avatar_path ? urls[r.avatar_path] : undefined} name={r.display_name || r.username} />
                <span className="min-w-0 flex-1 truncate"><strong>{r.display_name || r.username}</strong> <span className="text-muted">@{r.username} #{r.number}</span></span>
                <button onClick={async () => { await removeFriend(r.user_id); void refresh() }} className="min-h-10 px-2 text-sm font-semibold text-muted">Decline</button>
                <Button size="sm" onClick={async () => { await acceptFriend(r.user_id); void refresh() }}>Accept</Button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="mt-6 rounded-[24px] bg-surface p-4" aria-label="Leaderboard">
        <h2 className="m-0 mb-3 font-display text-base uppercase">Leaderboard</h2>
        <ol className="m-0 flex list-none flex-col gap-2 p-0">
          {board.map((p, i) => {
            const you = p.user_id === mine.user_id
            const a = activity(p.last_active)
            return (
              <li key={p.user_id}>
                <Row rank={i + 1} name={you ? `${p.display_name || p.username} (you)` : p.display_name || p.username} number={p.number}
                  sub={you ? 'You' : a.label} online={!you && a.online} avatar={p.avatar_path ? urls[p.avatar_path] : undefined}
                  streak={you ? streak.days : p.streak} lit={you ? streak.today : p.trained_today} rating={p.rating}
                  onOpen={() => setOpen(p)} highlight={you} />
              </li>
            )
          })}
        </ol>
        {board.length <= 1 && <p className="m-0 mt-3 text-sm text-muted">Add friends below: they show up here, ranked by streak, then rating.</p>}
      </section>

      <AddFriend onChanged={refresh} />

      {outgoing.length > 0 && (
        <section className="mt-4" aria-label="Sent requests">
          <h3 className="m-0 mb-2 font-display text-sm uppercase text-muted">Waiting for them to accept</h3>
          <ul className="m-0 flex list-none flex-col gap-2 p-0">
            {outgoing.map((r) => (
              <li key={r.user_id} className="flex items-center gap-3 rounded-2xl bg-surface px-3 py-2">
                <Avatar url={r.avatar_path ? urls[r.avatar_path] : undefined} name={r.display_name || r.username} />
                <span className="min-w-0 flex-1 truncate">@{r.username} <span className="text-muted">#{r.number}</span></span>
                <button onClick={async () => { await removeFriend(r.user_id); void refresh() }} className="min-h-10 px-2 text-sm font-semibold text-muted">Cancel</button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {open && <FriendSheet p={open} photo={open.avatar_path ? urls[open.avatar_path] : undefined} you={open.user_id === mine.user_id}
        onClose={() => setOpen(null)} onRemove={async () => { if (!confirm(`Remove ${open.display_name || open.username} from your friends?`)) return; await removeFriend(open.user_id); setOpen(null); void refresh() }} />}
    </Shell>
  )
}

function Row({ rank, name, number, sub, streak, lit, rating, avatar, online, onOpen, highlight }: {
  rank: number; name: string; number: number; sub: string; streak: number; lit: boolean; rating: number | null
  avatar?: string; online?: boolean; onOpen?: () => void; highlight?: boolean
}) {
  const body = (
    <>
      <span className="w-5 shrink-0 text-center font-display text-lg text-muted">{rank}</span>
      <span className="relative shrink-0">
        <Avatar url={avatar} name={name} />
        {online && <span className="absolute -right-0.5 -bottom-0.5 h-3.5 w-3.5 rounded-full border-2 border-surface-2 bg-[#2fbf71]" aria-label="Online" />}
      </span>
      <span className="min-w-0 flex-1 text-left">
        <span className="block truncate font-semibold">{name} <span className="text-muted">#{number}</span></span>
        <span className={`text-xs ${online ? 'text-[#2fbf71]' : 'text-muted'}`}>{sub}</span>
      </span>
      <span className="flex items-center gap-1 font-display tabular-nums" aria-label={`Streak ${streak}`}>
        <FlameIcon lit={streak > 0} className={`h-5 w-5 ${lit ? 'drop-shadow-[0_0_5px_rgba(255,90,31,.7)]' : ''}`} />{streak}
      </span>
      <span className="w-9 shrink-0 text-right font-display text-lg tabular-nums" aria-label={`Rating ${rating ?? 'none'}`}>{rating ?? '–'}</span>
    </>
  )
  const cls = `flex w-full items-center gap-3 rounded-2xl px-3 py-2.5 ${highlight ? 'bg-surface-2 ring-1 ring-red/40' : 'bg-surface-2'}`
  return onOpen ? <button onClick={onOpen} className={`${cls} active:scale-[0.99]`}>{body}</button> : <div className={cls}>{body}</div>
}

function Avatar({ url, name, big = false }: { url?: string; name: string; big?: boolean }) {
  const size = big ? 'h-16 w-16 text-xl' : 'h-10 w-10 text-sm'
  return url
    ? <img src={url} alt="" className={`${size} shrink-0 rounded-full object-cover`} />
    : <span className={`${size} grid shrink-0 place-items-center rounded-full bg-surface font-display uppercase text-muted`} aria-hidden="true">{name.slice(0, 1)}</span>
}

function AvatarPicker({ url, name, onChanged }: { url?: string; name: string; onChanged: () => void }) {
  const [busy, setBusy] = useState(false)
  const [cropping, setCropping] = useState<Blob | null>(null)
  const upload = async (f: Blob) => {
    setBusy(true)
    const ok = await setAvatar(f)
    setBusy(false)
    if (ok) onChanged(); else alert("Couldn't upload the photo. Check your connection and try again.")
  }
  return (<>
    {cropping && <CropSheet file={cropping} title="Profile picture" size={512} onCancel={() => setCropping(null)} onDone={(b) => { setCropping(null); void upload(b) }} />}
    <label className="relative shrink-0 cursor-pointer" aria-label="Change your profile picture">
      <Avatar url={url} name={name} big />
      <span className="absolute -right-1 -bottom-1 grid h-6 w-6 place-items-center rounded-full bg-red text-xs text-on-red">{busy ? '…' : '+'}</span>
      <input type="file" accept="image/*" className="sr-only" onChange={async (e) => {
        const f = e.target.files?.[0]
        e.target.value = ''
        if (f) setCropping(f)
      }} />
    </label>
  </>)
}

function UsernameForm({ current, profile, onDone, onCancel }: { current?: string; profile: Profile; onDone: () => void; onCancel?: () => void }) {
  const [name, setName] = useState(current ?? '')
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const valid = USERNAME.test(cleanUsername(name))
  return (
    <form className="mt-4 rounded-[24px] bg-surface p-4" onSubmit={async (e) => {
      e.preventDefault()
      if (!valid) return
      setBusy(true)
      const r = await claimUsername(name, profile)
      setBusy(false)
      if (r === 'ok') onDone()
      else setMsg(r === 'taken' ? 'That username is taken.' : r === 'too-soon' ? 'You can change your username once every 30 days.' : r === 'invalid' ? '3–20 letters, numbers, dots or underscores.' : "Couldn't save. Check your connection.")
    }}>
      <h2 className="m-0 font-display text-base uppercase">{current ? 'Change your username' : 'Pick your username'}</h2>
      <p className="m-0 mt-1 text-sm text-muted">Friends find you with your username and shirt number (#{profile.number}).{current ? '' : ' You can change it later.'}</p>
      <label htmlFor="team-username" className="mt-4 block text-sm text-muted">Username</label>
      <div className="mt-1 flex items-center rounded-2xl bg-surface-2 px-3">
        <span className="text-muted">@</span>
        <input id="team-username" value={name} onChange={(e) => { setName(e.target.value); setMsg(null) }} autoCapitalize="none" autoCorrect="off" spellCheck={false}
          placeholder="e.g. peter.7" className="min-h-12 min-w-0 flex-1 bg-transparent px-1 text-[16px] text-ink placeholder:text-muted" />
      </div>
      <p className={`m-0 mt-1 text-sm ${msg ? 'text-red-ink' : 'text-muted'}`}>{msg ?? '3–20 letters, numbers, dots or underscores.'}</p>
      <div className="mt-4 flex gap-2">
        {onCancel && <Button type="button" variant="ghost" className="flex-1 bg-surface-2" onClick={onCancel}>Cancel</Button>}
        <Button type="submit" className="flex-1" disabled={!valid || busy}>{busy ? 'Saving…' : 'Save'}</Button>
      </div>
    </form>
  )
}

function AddFriend({ onChanged }: { onChanged: () => void }) {
  const [name, setName] = useState('')
  const [num, setNum] = useState('')
  const [found, setFound] = useState<FoundPlayer | null | undefined>(undefined) // undefined: not searched
  const [url, setUrl] = useState<string | undefined>()
  const [busy, setBusy] = useState(false)
  const search = async () => {
    setBusy(true)
    try {
      const f = await findPlayer(name, Number(num))
      setFound(f)
      setUrl(f?.avatar_path ? (await avatarUrls([f.avatar_path]))[f.avatar_path] : undefined)
    } catch { setFound(null) }
    setBusy(false)
  }
  const field = 'min-h-12 min-w-0 rounded-2xl bg-surface-2 px-3 text-[16px] text-ink placeholder:text-muted'
  return (
    <section className="mt-4 rounded-[24px] bg-surface p-4" aria-label="Add a friend">
      <h2 className="m-0 font-display text-base uppercase">Add a friend</h2>
      <p className="m-0 mt-1 text-sm text-muted">Their username and shirt number, both exact.</p>
      <form className="mt-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); if (name && num) void search() }}>
        <input aria-label="Friend's username" value={name} onChange={(e) => { setName(e.target.value); setFound(undefined) }} placeholder="@username"
          autoCapitalize="none" autoCorrect="off" spellCheck={false} className={`${field} flex-1`} />
        <input aria-label="Friend's shirt number" value={num} onChange={(e) => { setNum(e.target.value.replace(/\D/g, '').slice(0, 2)); setFound(undefined) }}
          placeholder="#" inputMode="numeric" className={`${field} w-16 text-center`} />
        <Button type="submit" size="sm" disabled={!name || !num || busy}>Find</Button>
      </form>
      {found === null && <p className="m-0 mt-3 text-sm text-muted">No player with that username and number.</p>}
      {found && (
        <div className="rise mt-3 flex items-center gap-3 rounded-2xl bg-surface-2 p-3">
          <Avatar url={url} name={found.display_name || found.username} big />
          <span className="min-w-0 flex-1">
            <strong className="block truncate">{found.display_name || found.username}</strong>
            <span className="text-sm text-muted">@{found.username} · #{found.number}</span>
          </span>
          {found.friendship === 'friends' ? <span className="text-sm text-muted">Friends ✓</span>
            : found.friendship === 'requested' ? <span className="text-sm text-muted">Requested</span>
              : found.friendship === 'asked-you'
                ? <Button size="sm" onClick={async () => { await acceptFriend(found.user_id); setFound({ ...found, friendship: 'friends' }); onChanged() }}>Accept</Button>
                : <Button size="sm" onClick={async () => { if (await askFriend(found.user_id)) { setFound({ ...found, friendship: 'requested' }); onChanged() } }}>Add</Button>}
        </div>
      )}
    </section>
  )
}

// A friend's player card, from what they published.
function FriendSheet({ p, photo, you, onClose, onRemove }: { p: TeamProfile; photo?: string; you: boolean; onClose: () => void; onRemove: () => void }) {
  const profile: Profile = {
    name: p.display_name || p.username, number: p.number, positions: p.positions as Position[],
    foot: p.foot as Foot, accent: p.accent as Accent, idol: '',
  }
  const a = activity(p.last_active)
  return (
    <Sheet label={`${profile.name}'s card`} onClose={onClose}>
        <div className="mx-auto w-full max-w-[260px]"><PlayerCard p={profile} stats={p.card} photo={photo} /></div>
        <div className="mt-4 grid grid-cols-3 gap-2 text-center">
          <Stat label="Streak" value={`${p.streak}`} />
          <Stat label="Best" value={`${p.best_streak}`} />
          <Stat label={you ? 'You' : a.online ? 'Online' : 'Seen'} value={you ? '—' : a.online ? '●' : a.label} />
        </div>
        {!p.card && <p className="m-0 mt-3 text-center text-sm text-muted">Not rated yet.</p>}
        <div className="mt-5 flex gap-3">
          {!you && <Button variant="ghost" className="flex-1 bg-surface-2" onClick={onRemove}>Remove friend</Button>}
          <Button className="flex-1" onClick={onClose}>Close</Button>
        </div>
    </Sheet>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-surface-2 px-2 py-3">
      <span className="block font-display text-lg tabular-nums leading-tight">{value}</span>
      <span className="text-xs text-muted">{label}</span>
    </div>
  )
}
