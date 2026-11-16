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
