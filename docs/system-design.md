# System design: Football Personal Coach

Draft for discussion (2026-11-16). This is where the app is going so Team, the shared exercise library and notifications can be built without rewrites.

## What the app is for

- **Main demand:**
  - plan the week around the game
  - correct instructions for every exercise
  - detailed football sessions
- **What brings users in:**
  - a friends leaderboard (streaks, ratings, PRs)
  - the player card rating
  - training with friends

## Today

| Part | How it works now |
|---|---|
| App | React PWA on Cloudflare. Local-first: everything is saved on the phone first, then synced. |
| Accounts | Supabase email + password. |
| Data | One private table, `user_state`: one JSON value per user per key. The keys are `history`, `games`, `settings`, `quotes`, `mySessions`, `profile`, `weekPlan` and `videos`. Row-level security lets each user see only their own rows. |
| Exercise library | `web/src/data/library.json`, shipped inside the app. |
| Photos | Private Supabase Storage folder per user. |

This works for one person's own data. It **can't do Team**:
- Nobody can see anyone else's data.
- Nothing can be searched (usernames).
- Streaks and ratings are worked out on the phone, so a leaderboard could be faked.

So Team needs a redesign of the data, not just new screens.

## The design: shared vs private

The rule: **whatever other people see lives in real tables and is checked or calculated by the server.** Private things can stay simple.

### 1. Exercise library (shared, everyone reads it)

`exercises`

| Field | |
|---|---|
| id, slug, name | |
| `sport` / `place` | pitch, gym, home |
| `category` | lower body, plyometrics, ball mastery… |
| `type` | strength, power, skill, conditioning, mobility |
| `equipment[]` | barbell, box, ball, wall, sofa… |
| `muscles[]` | |
| `positions[]` | |
| Defaults | sets, reps, work time, rest |
| `how` | Step-by-step instructions |
| `cues` | Coaching points, tempo |
| `video` | Licensed video (our own recordings or 3D renders), stored in Storage |
| `status` | draft / submitted / published / rejected |
| `created_by`, `version`, `updated_at` | |

- **Contributors** submit exercises with status *submitted*. An admin or moderator publishes them, and only published ones are used by the planner.
- **Personal video links** (the TikTok links in today's app) stay private per user, because creators own those videos. The shared library only carries videos we have the right to show.

`session_templates` holds recommended sessions and users' own sessions:
- items that point at exercise ids, plus per-item overrides and superset flags
- a `public` flag, so a user can share a session

### 2. Users

| Table | Who sees it | What |
|---|---|---|
| `profiles` | Friends (and search) | `username` (unique), display name, shirt number, positions, avatar, team colour, `last_active` |
| `player_private` | Only you | Injuries, body stats, preferences, games, week changes. Most of today's `user_state` can stay here. |
| `ratings` | Friends see the card | One row per rating session: date, test, result. The card is calculated from these. |
| `rating_rules` | Admin | How results turn into card numbers (thresholds per test, version). Tuning it never touches user data. |
| `workouts` | Only you; friends see the totals | One row per session: date, template, type, minutes, effort, load, deleted. Replaces the `history` JSON. |
| `personal_records` | Friends | Exercise, value, unit, date. Optionally linked to a workout. |

### 3. Team

| Table | What |
|---|---|
| `friendships` | requester, addressee, status (pending / accepted / blocked). Add by username + shirt number; you see their card before adding. |
| `leaderboard` (view) | Per user: current streak, best streak, overall rating, top PRs, last active. A friend can only read rows of accepted friends. |

The **streak is calculated by the server** from `workouts`, using the same rules as `lib/streak.ts`. The leaderboard can then be trusted.

### 4. Notifications

| Table | What |
|---|---|
| `push_subscriptions` | One per device: endpoint and keys. |
| `notification_prefs` | Sunday plan reminder on/off, time, timezone. |

- A scheduled job (Supabase cron + an Edge Function) runs every hour. It sends "Plan next week" to anyone whose local time is Sunday 18:00.
- On iPhone, push only works when the app is added to the Home Screen (iOS 16.4+).

### 5. Admin

- `roles`: admin, moderator.
- `audit_log`: who changed what.
- `reports`: flagged exercises or users.
- Start with the Supabase dashboard. Later, a small admin page inside the app that only the admin role can open.

## What keeps it easy to change

- **Every database change is a numbered migration file in the repo**, tested locally first. That's already how we work.
- **Generated TypeScript types** from the database schema (`supabase gen types`), so a schema change shows up as a type error in the app.
- **Rules live in data, not code, where they change often:** rating thresholds and the exercise library.
- **Calculations that others see run on the server:** streaks and leaderboards.
- **Local-first stays** for your own data, so the app works offline in the gym or on the pitch.
- **A feature flags table**, to turn Team on for testers first.

## Suggested order

1. **Usernames + public profiles + friends + leaderboard.**
   - Includes moving `history` into `workouts` (needed for server-side streaks).
   - Biggest pull for new users.
2. **Personal records + rating history.** Compete on PRs and ratings.
3. **Notifications.** Sunday plan reminder.
4. **Shared exercise library in the database.**
   - Contributions + moderation.
   - Then licensed videos (record our own, or 3D).
5. **Train with a friend.** A shared live session at the gym or pitch, on the same timer.

## Decisions (2026-11-16)

- **Visibility:** card, streak and rating are visible to **friends only**. Search shows only username, name, number and photo, and needs the exact username **and** shirt number.
- **Leaderboard:** friends first. Teams and clubs come later on the same tables.
- **PRs:** self-reported for now. A friend's "I was there ✓" comes later.
- **Usernames:** unique and changeable. The first change is free, then once every 30 days.
- **Quotes:** kept, never with a name underneath.
- **RAG scripts:** kept, as the portfolio piece and the reference for the AI Coach (Claude + retrieval from our own database).

## Phase 1: built (migration 0005)

- `profiles`, `friendships`, `find_player()`, `friend_requests()` and the `avatars` bucket. Tested in `supabase/tests/team_test.sql`.
- **Change from the plan:** the training log stays in `user_state`. The app publishes streak, rating and last active to `profiles`, and the database checks the ranges.
  - Server-side streaks would mean rewriting the week planner in SQL or running an Edge Function.
  - It wouldn't stop fake logs anyway.
  - The log moves to a `workouts` table with phase 2 (PRs) and the AI Coach.
- **Online** means active in the last 5 minutes. The app publishes on open and every 3 minutes while it's open.
## Open questions

- **Who can see your card and streak:** friends only, or anyone who searches your username?
- **Username rules:** unique, changeable?
- **Leaderboard groups:** friends only, or also teams/clubs (your futsal team)?
- **PRs:** self-reported, or need a video or a friend to confirm?
