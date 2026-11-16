# Training references

Pro and coach examples Peter sends (TikTok and others). For each: what it shows, what we learned, what changed in the app. Only notes and links go here, never their videos.

## 1. A pro footballer's gym week (Sarmiento, TikTok, 2026-11-16)

Game on Saturday. His gym days:

| Day | Matchday | Job |
|---|---|---|
| Mon | MD+2 | Active recovery (foam roller, mobility) |
| Tue | MD-4 | Strength / contrast, lower body (heavy squat + box jumps) |
| Wed | MD-3 | Core / injury prevention (plate halos, half-kneeling rotation) |
| Thu | MD-2 | Submax strength / power, upper body (bench press) |
| Fri | MD-1 | Isometric / neural activation (hip thrust holds, short fast work) |
| Sat | MD | Game |

**What we learned:**
- "Light" doesn't mean the same session with fewer sets. Each day has its own job, set by its distance to the game.
- Legs go far from the game.
- The day before the game is a short activation, not rest and not upper body.
- Core and injury prevention get a day of their own.

**What changed (planner v4, `web/src/lib/weekplan.ts`):**
- Gym days get a job by matchday:
  - MD-4 or more: legs / contrast
  - MD-3: core & injury prevention
  - MD-2: upper body power
  - MD-1: activation
  - MD+1: recovery
  - MD+2: only light gym if needed
- The planner fills the most useful jobs first. With 3 gym sessions that's legs, upper and core; a 4th adds activation.
- New sessions:
  - Gym: upper body power (submax)
  - Gym: core & injury prevention
  - Gym: match-eve activation (~23 min)
- Football goes on MD-3/-4 (high intensity), MD-2 (sharp passing) or further out (technique). Never MD-1.
- Matchday labels wrap around the week and follow football convention: MD+1 and MD+2, then counting down to the next game.

**Open:** his exercises we don't have yet: plate halo, half-kneeling rotations, isometric hip thrust hold, foam roller routine.

## 2. "How to train weekly like a Premier League player" (@ballersmindset_2 photos; video @effordelite, 2026-11-16)

Game on Saturday:

| Day | Matchday | Job |
|---|---|---|
| Mon | MD-5 | Upper body gym + ball work |
| Tue | MD-4 | Team training + legs + speed |
| Wed | MD-3 | Mobility / light session, or full rest |
| Thu | MD-2 | Team training |
| Fri | MD-1 | Tactical walk-through or light gym |
| Sat | MD | Matchday |
| Sun | MD+1 | Recovery or rest |

**Agrees with reference 1:** legs at MD-4, a light day before the game, recovery after it. Gym and ball work on the same day is normal.

**New lesson: hard days hard, easy days easy.** MD-4 stacks football, legs and speed on one day, and MD-3 after it is deliberately easy.

**What changed:**
- The high-intensity football session now goes **with** the MD-4 legs day (ball work and sprints first, gym after) instead of avoiding gym days.
- MD-3 has no football. Its gym job (core & injury prevention) is low load.
- Second football session: MD-2 or technique further out, as before.

**Not taken:** upper body at MD-5. Reference 1 puts upper at MD-2, and both work; we keep MD-2 so it stays away from the legs day.

The video caption couldn't be read (TikTok blocks fetching it). Paste the text if it adds anything.
