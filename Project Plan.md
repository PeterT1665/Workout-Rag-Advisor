# Sport Personal Coach — Project Plan (v2)

_Last updated: 2026-11-08 (v3 direction). v2 started 2026-10-16; the original CLI-only plan is in git history._

## 1. What this is

**v3 (agreed 2026-11-08):** a training app for **football players who also train in the gym** to perform better (strength, explosiveness, endurance). Cheerleading is dropped. Built for Peter first, then 3–5 invited friends, designed so it can grow into a startup without a rewrite.

- **One app, two ways in.** *Try it*: guest mode, no account, data stays in the visitor's browser, sample player or their own player card (portfolio demo, coffee chats, interviews). *Sign in*: a private account, synced between phone and laptop.
- **Each user customises their app:** position(s) (winger, striker, midfielder, defender; several allowed), dominant foot, game days, theme colour, their own idol (name, photos, clips, quotes, stored privately).
- **Phone-first.** Planning and training both work fully on the phone; the laptop layout is the same app on a wide screen.
- **Exercise library + sessions.** Every drill/exercise lives once in a library, grouped by category; sessions are built from it, either recommended templates (per position, like Upper/Lower or Push/Pull/Legs for gym) or the user's own.
- **Later:** a Coach agent (Gemini/Groq) on top of the library and the user's data; native iPhone app.

Secondary purpose: a portfolio project. Peter must be able to explain RAG, the agent design and the trade-offs in an interview (see section 9).

## 2. Current state (honest)

Built, CLI only, gym only:
- `ingest.py` — indexes 873 exercises + 3 personal files (876 docs) into a LlamaIndex simple vector store (folder is named `chroma_db` but it is **not** Chroma).
- `query.py` — semantic retrieval + Python post-filtering (injury/plyo, equipment).
- `planner.py` — 7-day gym template, grounded prompt, Ollama generation.
- `main.py` — typer CLI: `generate`, `query`.

Not done: Ollama never installed, so no plan has been generated end to end. Prompt tuning is outstanding (Thursday pull day is biceps-heavy; Saturday core day pulls chest exercises). No football or cheerleading content exists. No web UI, phone app, agent, or hosting.

## 3. Decisions made

| # | Decision | Notes |
|---|---|---|
| D1 | Phone app is a **PWA** (installed to home screen) | Screen stays on during drills. Native app is a possible later upgrade if background audio/timers prove unreliable. |
| D2 | LLM: **Gemini free tier primary, Groq free tier fallback** | Local Ollama is optional offline backup, not a requirement. |
| D3 | Hosting: **AWS EC2** (Peter has experience) | Check current free-tier terms/cost before committing. |
| D4 | **Data stays on our own server** (SQLite on the EC2 box), editable through the app | See section 6. Only the minimum context is sent to Gemini/Groq. |
| D5 | **Football first**, then gym, then cheerleading | |
| D6 | Reminders = **calendar export (.ics)** | Push notifications deferred. |
| D7 | Voice cues: **random timing, "left" / "right" only, 1v1 drills only** | Spoken aloud (phone speaker or earphones). Richer cues later, after a working prototype. |
| D8 | Gym music / motivational speeches: **noted, not built** | Backlog. |
| D9 | **Weekly plan flexes with the number of games.** Football sessions: **1 game → 2, 2 games → 1, 3 games → 0** (gym + recovery instead) | See section 5.0. Hard weeks get recovery built in; gym is not just "more work". Confirmed by Peter 2026-10-23. |
| D10 | Hosting: any free-tier option is fine; **Docker on EC2 is the default** | Peter's AWS account is still in its free period and he has deployed one project there. Not tied to AWS. |
| D12 | No physio check for now; **keeping the gym membership for now** | Revisit later; not urgent. |
| D13 | **Design:** black, white and red; simple. Cristiano Ronaldo (Portuguese footballer) is the role model, red as the main colour | No logos or trademarks used; palette only. Peter can connect a Figma/web design later. |
| D14 | **Web search via the LLM API** (Gemini / Groq), free tier | See section 7 note. Verify free-tier limits before relying on it. |
| D15 | **Mindset clips:** Peter's own Ronaldo audio clips (commentary, interviews) play before the session and during rests; SIUUU clip at the end; motivational cards on start/rest/finish screens | Real recordings Peter supplies, stored only on his phone (IndexedDB), never uploaded or committed. Not to be published (broadcaster copyright). No AI voice cloning of Ronaldo. Built 2026-10-23. |
| D16 | **Frontend is React + TypeScript + Tailwind (Vite), in `web/`.** Framer only for a later portfolio page | Approved 2026-10-24. Old plain-JS `app/` removed after full port. |
| D17 | **Design skills installed (project-level):** Impeccable (hooks off for now) + Emil Kowalski's design-eng, mobile-native, animate, review-animations, apple-design | Approved 2026-10-24. Feel: aggressive, modern premium, speedy. Design pass pending more feel answers. |
| D18 | **Design direction: game-like (FC27-inspired, not copied).** Full-screen red LEFT/RIGHT cue with giant arrow; full-screen "slam" transitions; dark + light modes, red kept in both (brighter in dark, deeper in light); abstract football textures (turf, chalk lines, grain); big "7" motif and Ronaldo named in quote attributions; big Today's session card on home; **player card** (ratings from Peter's own tests) as the progress screen | Agreed 2026-10-24. No EA logos/fonts/card designs; no "CR7" mark; no Ronaldo photos in the repo (Peter's own photos only on-device, like clips). Mockups in `design/mockups/`. Awaiting: pick mockup A/B (or mix) and font A/B. |
| D19 | **Product principles from Peter's app research (2026-10-24):** (1) **zero logging mid-workout**: the app records what it can automatically (time, drills done, cues called); (2) **post-session 20-second reflection** by taps (effort 1–10, legs, soreness) + optional voice note, no forms; (3) **visual, not lists of words**: rings, big numbers, stat tiles, drill diagrams (STATSports-style); (4) **matchday labels** (MD-2, MD+1) tie sessions to games; (5) **session load = minutes × effort (session-RPE)** drives the weekly plan; (6) player-card ratings come from occasional test days, not daily logging | Widgets: an iOS PWA cannot add home/lock-screen widgets (needs a native app). Interim: the exported calendar events show in iOS's Calendar widget as "next session". Native widget = later upgrade. Mockup set C in `design/mockups/`. |
| D20 | **Peter wants it to feel like a 3D game with animation**, not flat 2D (noted 2026-10-24, not yet designed) | Plan: real 3D only for big moments (player card reveal, kick-off intro, full-time); depth and motion elsewhere; the in-drill player stays fast and readable. Check battery use on the phone. |
| D21 | **Design picks (2026-10-24):** mockup **C** as base; **font B** (Archivo Expanded Black, wide & heavy); style = **"aggressive fun"**: low-poly 3D football objects (ball, cones, goal, boots, water) in a real 3D scene, tilted stickers, angled buttons, sunset-red sky. Same fun style later for cheerleading with its own objects | Reference: mockup E (`design/mockups/mockup-e-home.html`). 3D objects v2 are smooth/clean (D was too rough), built in code in `design/mockups/objects3d.js` (three.js 0.170), reusable in the app; no image licensing issues. |
| D22 | **Photos + 3D, rounded (2026-10-24):** photo backgrounds for atmosphere (red night goal, gear flat-lays, Ronaldo jersey), 3D gear objects on top (ball, cone, boots, water, disc markers, hurdle, speed rings, pole, goal); **rounded buttons/cards** (~16px buttons, ~20px cards, pills for tags), STATSports-style photo carousels, raised round Kick off button in the tab bar | Peter's Pinterest photos live in `design/private/` (git-ignored); swap for Unsplash/Pexels or Peter's own photos before any public/portfolio use. Mockup F = reference (`design/mockups/mockup-f-home.html`). |
| D23 | **Two builds, personal and public (2026-10-24).** Personal = Peter's private assets (Ronaldo photos, clips, Pinterest pictures) loaded from the device / `design/private/`, never committed. Public/portfolio = licensed or Peter's own photos only. **More Ronaldo in the personal version**: jersey hero, celebration panels in Mindset, jerseys behind the player card. Red sky and red-lit goal hero rejected. New reference photos used for inspiration only (`design/moodboard.md`). | Mockup G = current personal reference. |
| D11 | Cheap cones will be bought; bottles/shoes/bags used until then | Plan drills so they work with either. |

**v3 decisions (2026-11-08).** These replace D3, D4, D10 and the single-login part of section 6; D5 changes (cheer dropped); D1 gets a native follow-up; D23's two builds become one app with guest mode.

| # | Decision | Notes |
|---|---|---|
| D24 | **Audience:** footballers who also go to the gym (3–4+ gym sessions a week is normal). Football + gym only; **cheerleading dropped** | Gym exercises serve football performance and general fitness (strength, explosiveness, endurance). |
| D25 | **One app, guest mode + accounts** (replaces the separate demo build long term) | Guest data stays in the browser; sample player is an option. |
| D26 | **Backend: Supabase** (Postgres, auth, private file storage), **Sydney region**. Replaces D4 (own server) | Per-user access enforced in the database (row-level security). Standard Postgres, so it can be moved later. Free tier pauses after ~1 week idle. |
| D27 | **Website hosting: Cloudflare Pages**, deployed by **GitHub Actions** (tests on every push; deploy only on merge to `main` when `web/` changed). Replaces D3/D10 | Netlify rejected: free plan charges credits per production deploy. EC2 + DuckDNS retired once the new site is live. |
| D28 | **Sign-in: username + password**, email optional | Supabase needs an email, so usernames map to an internal address; no self-service reset without a real email (Peter resets in the dashboard). Invite-only: public sign-up off. |
| D29 | **Positions:** winger, striker, midfielder, defender; **multiple per player** | Session recommendations per position. |
| D30 | **Exercise library with categories; sessions composed from it** (recommended templates + user-built, editable on the phone) | Football: warm-up, ball mastery/dribbling, first touch, passing, shooting/finishing, 1v1, speed & agility, conditioning, cool-down/mobility. Gym: lower strength, upper strength, power/plyo, core, conditioning. |
| D31 | **Phone-first planning**; laptop = same app, wide layout | |
| D32 | **Native iPhone app via Capacitor** (wraps the React app), so the timer and voice keep running with the phone locked | Free Apple account = Peter's phone only, reinstall every 7 days; Apple Developer Program (US$99/yr) when friends need it (TestFlight). PWA stays for everyone else. |
| D33 | **Week planner must handle 3–4+ gym sessions** alongside games and football sessions | Open: on heavy game weeks, lighten gym (default proposal), cut sessions, or only warn. |

## 4. User profile

Kept private in `data/profile.md` (git-ignored). See `data/examples/` for the shape of the personal files.

## 5. Product scope

### 5.0 Flexible weekly load (applies to everything)

The number of games changes week to week (1, 2 or 3), so the plan is built **around the week's games, not a fixed template**. Each week starts with Peter telling the agent (or the app asking) which games are coming, their day and type.

Typical game pattern:
- Competitive futsal: usually **Wednesday night**, 40 min.
- Casual game: usually **Monday or Wednesday**, 90 min – 2 h. Sometimes two games on Wednesday, sometimes no competitive futsal that week.
- Sunday game: very rare.
- Cheerleading: Saturday morning, fixed.

Rules the planner follows (to be refined with Peter):
- **Game count sets the load.** 1 game = room for 3 football sessions and some gym power work. 3 games = 1–2 lighter football sessions, no extra hard work.
- **Games count as hard days.** Put hard training away from games (no heavy legs or sprint work the day before; light the day after).
- **Hard weeks include recovery:** mobility/hip session, light technical football (ball mastery, wall passing at low intensity), or a recovery-focused gym session. Gym is *not* simply more work to boost performance.
- **Soreness feedback** (from the log) can reduce the next session's volume.
- **past hip/knee overuse guardrail:** hip/glute strengthening and hamstring/quad work stay in every week; no sudden jumps in volume.
- Friend sessions are slotted in when Peter says a friend is available.

The agent explains why it shaped the week that way ("3 games this week, so only one light skills session and one recovery session").

### 5.1 Football (priority 1)
Self-training that mirrors game scenarios for an attacker.

**Session building blocks** (agent composes 1–1.5 h sessions from these):
1. Warm-up without the ball (dynamic mobility, hips/glutes/IT-band-friendly activation)
2. Ball control: juggling (both feet, thighs, head)
3. Sprints and change-of-direction (no cones)
4. Close control in tight space, under pressure
5. Dribbling: driving the ball, changing direction, freestyle
6. Shooting: stationary, on the move, from dribble; weak-foot (left) focus
7. 1v1: beat a "defender" then shoot / cross / pass — **voice cue drills**
8. Wall passing: on the ground and in the air, both feet
9. Passing/crossing drills
10. Cool-down and mobility

**Interactive timer (phone):**
- Background interval timer: work/rest, "next exercise" call-outs.
- Voice cues on 1v1 drills only: random-interval "left" / "right" (speech synthesis in the browser). Frequency and gap range configurable.
- Later ideas (not now): richer cues, difficulty ramps, cue sets per drill.

**Position research** (agent's first job): what a match demands from a winger / striker / CM in 5-a-side vs 7-a-side (movement, sprint distances, actions per game, typical situations), turned into training priorities. The agent should look this up itself and ask Peter only for what it cannot find.

**Match reflection:** after each game, Peter logs how it went (minutes, position, what worked, what didn't, fatigue/soreness); the agent feeds this into next week's plan.

**Sessions with a friend** (housemate, occasionally): mostly **one- and two-touch passing** for ball mastery, in the air (control) and on the ground (accuracy). Sometimes tactical work or 1v1. The agent has a "with a partner" session type built from these.

### 5.2 Gym (priority 2)
Weight training for athleticism and explosive power, supporting football speed/agility and cheerleading. Existing exercise RAG is reused. Must include **hip mobility / glute strength work** because of the past hip/knee overuse history. Heavy and explosive lifts for football speed need gym equipment, so the plan should be purposeful (specific exercises for sprint speed, acceleration, change of direction), not general.

Peter may cancel the gym membership if it turns out he doesn't need it. The app should make this decision easier by showing which planned exercises genuinely need gym equipment.

### 5.3 Cheerleading (priority 3)
Weekly Saturday-morning session: proper warm-up, skill progression within the session, freedom to try new skills by capability. A simple plan, not deep.

### 5.4 Backlog (not now)
Music / motivational speeches in gym sessions; push notifications; native iOS app; richer voice cues; defender-style audio scenarios.

## 6. Data and privacy design

**v3 (D26, D28):** each user's data lives in Supabase (Sydney), readable only by that user (row-level security). Idol photos/clips go in private per-user storage, never shared, never used in marketing; the demo keeps licensed photos only. Health data (injuries, soreness, body stats) is sensitive: invite-only sign-up, a delete-my-account option and a short privacy note now; a proper privacy policy and terms (Australian Privacy Act) before any public launch. Free Gemini/Groq tiers may train on what is sent, so for other users strip details or get consent before the Coach agent sends their data.

_v2 design below (single user, own server), kept for history:_

- All personal data (profile, body stats, PRs, injuries, match logs, plans, session history) lives in **SQLite on our own server**, not in a third party.
- The web app has **forms to update** body weight, body fat, PRs, injuries and match logs — not only by editing files. The agent can also update them from chat ("log: weight 67.5 kg").
- The data files in `data/` become seed/export formats, not the source of truth.
- Gemini/Groq only receive the context needed for a given question (e.g. current goals, active injuries, this week's load), not the whole database.
- **Caveat to verify before relying on it:** free API tiers often allow the provider to use submitted content to improve their models. Check Gemini's and Groq's current terms. If that is unacceptable for body/injury data, strip or generalise it before sending (e.g. "recent knee overuse injury" rather than details), or use the paid tier.
- Access control: a single login (password or token) in front of the server, HTTPS, regular backups of the SQLite file.

## 7. Architecture (target)

**v3:**
```
iPhone (PWA now, Capacitor app later) + laptop browser: one React app
              │ static files                         │ HTTPS (auth, data, uploads)
   Cloudflare Pages ◄── GitHub Actions          Supabase (Sydney)
   (deploy on merge to main)                     Postgres + RLS, Auth, Storage
                                                     │ later
                                             Coach agent + RAG (where it runs: TBD;
                                             TypeScript function, hosted Python, or small server)
```

_v2 architecture (FastAPI + SQLite on EC2), kept for history:_

```
iPhone PWA (session view, timer, voice cues, .ics export)
Laptop web app (agent chat, plan, progress, data forms)
              │  HTTPS
        Backend API (Python, FastAPI)           ← on AWS EC2
              │
   ┌──────────┼──────────────────────┐
 SQLite   Agent loop              Knowledge (RAG)
 (user    (tools: search web,     exercise DB + football drill/
  data)    retrieve, read/write    position guides, vector index
           user data, escalate)
              │
        Gemini (primary) → Groq (fallback)
```

**Web search:** both providers can search through the API instead of a separate search service. Gemini offers a Google Search grounding tool; Groq offers models with built-in web search. Free-tier limits and availability change, so check before building on them. If limits bite, the fallback is a free search API (e.g. a few hundred queries/month) behind the same tool interface.

The agent decides per question: answer from stored knowledge (RAG), look up fresh info on the web, or read/update Peter's data. Existing Python code (ingest/query/planner) becomes the knowledge and planning modules behind the API.

## 8. Build order

**v3 phases:**
1. **Pipeline + hosting:** GitHub Actions (tests, deploy), Cloudflare Pages, Supabase project. Then move the portfolio link, retire EC2 (terminate instance, release the Elastic IP) and the DuckDNS names.
2. **Player profile + onboarding + guest mode:** positions (multi), foot, game days, colour theme, idol; the app reads everything from the profile instead of Peter's hard-coded details.
3. **Exercise library + session builder:** categories, recommended templates per position, user-built sessions on the phone; gym library and templates.
4. **Accounts:** Supabase username/password sign-in, per-user data and sync, private idol uploads, Peter's data moved over. Invite 3–5 friends.
5. **Week planner v3:** 3–4+ gym sessions with games and football (D33).
6. **Native iPhone app (Capacitor):** background timer and voice.
7. **Coach agent** (Gemini/Groq) + football research.
8. **Startup readiness:** privacy policy, terms, analytics, error monitoring, payments if needed.

_v2 build order, kept for history:_

**Phase 0 — Decisions & research (no code)**
- Close the open questions in section 10.
- Verify current Gemini/Groq free-tier limits and data terms; choose a web-search tool for the agent.
- Draft one real football week on paper from research; Peter reviews. If the plan isn't good, no app fixes it.

**Phase 1 — Football brain (backend)**
- Swap Ollama for Gemini with Groq fallback behind one LLM interface.
- Football knowledge base (drills, position profile) + web research tool.
- Data model (SQLite) and profile/injury/match-log entities.
- Football weekly plan + session generator that respects load limits and the past hip/knee overuse history.

**Phase 2 — Laptop web app**
- Agent chat, weekly plan view, data forms, progress view.

**Phase 3 — Phone PWA**
- Session view, interval timer, spoken 1v1 cues (random left/right), keep-screen-awake, `.ics` calendar export.
- Prototype early and test the voice/timer on the actual phone on the field.

**Phase 4 — Deploy** to EC2 (HTTPS, login, backups).

**Phase 5 — Gym** (reuse existing RAG, add football-aware weekly balancing, fix prompt-tuning issues).

**Phase 6 — Cheerleading** session template.

**Later:** backlog in 5.4.

## 8b. Build status (2026-10-23)

Done (tested):
- `weekplan.py` — week skeleton from game count and days (rules in 5.0). 7 unit tests in `tests/test_weekplan.py`. Try: `python3 weekplan.py 2:competitive 0:casual`.
- `web/` (React, replaced `app/` 2026-10-24) — iPhone-ready PWA session player: timer, spoken announcements, random LEFT/RIGHT cues on 1v1 drills only, pause/skip, wake lock, calendar (.ics) export, offline cache, settings. 17 tests (logic, player engine, full-app UI flows) plus screenshots checked in real Chrome at iPhone size. See `web/README.md`.
- Mindset clips + SIUUU + motivational cards (D15), managed in Settings → Clips & quotes; covered by an extra simulated-browser test.
- `web/src/data/sessions.json` — 4 **draft** sessions (close control & 1v1, wall/weak-foot/shooting, partner passing, hip recovery).

Not tested on a real iPhone yet: speech/sound with screen on and off, wake lock (needs HTTPS), install to home screen.

**2026-10-24: mockup G built into the app (`web/`):**
- Home: walk-out hero, matchday tag (MD-2 etc.), today's session from the week plan, Kick off, kit check (real photo crops, tap to pack), today's drill cards, week strip, Mindset carousel (Ronaldo panels + quotes + clip button), player card tile.
- Tabs: Home / Week (tap game days, plan rebuilds with reasons) / raised ball button = Kick off / Player (card + history) / Coach (placeholder).
- Week planner ported to TypeScript (`web/src/lib/weekplan.ts`, same 7 cases as `weekplan.py`).
- Full-screen red LEFT/RIGHT cue with slam animation; font B (Archivo expanded); rounded UI.
- Personal photos load from `web/public/personal/` (git-ignored) with fallbacks; `npm run build:public` strips them for the public/portfolio build.
- 31 tests. Code-made 3D objects dropped: Peter found them not real enough. Kit uses real photo crops; real 3D models or Peter's own gear photos later.

**2026-10-24 (later):** laptop layout. Wide screens (≥1024px) get a sidebar (Home, Week, Sessions, Player, Coach, Kick off, Settings) and a planning dashboard (hero + week-at-a-glance list, drills, kit, player card, mindset). The phone keeps the tab bar and in-session experience. Same app, same URL. Kick-off motion, screen transitions, Impeccable checker clean, 33 tests.

**2026-10-25 polish pass:** full-time screen + 20-second check-in (effort 1-10, legs, soreness), session load = minutes x effort (session-RPE) saved to history; weekly load chart (Player tab + laptop dashboard); session details with match timeline and phase groups; segmented progress + up-next card in the player; new app icon (red 7); split red into fill (#e01b35, white text 4.8:1) and text (#ff2d46) tokens; Impeccable detector 0 findings; 36 tests.

**2026-10-25 21st.dev:** MCP connected (free tier: 2 component downloads/day). Used "Animate Digits" (unlumen) for the session timer and full-time load, adapted in `web/src/components/AnimateDigits.tsx`; `motion` library added; spring-sliding active indicators in the tab bar and sidebar.

**2026-10-26 deploy prep:** decisions: everything on EC2 (Caddy auto-HTTPS + free DuckDNS address); personal version online behind a password login; repo stays public with personal data moved out of git (`data/examples/` templates; `data/profile.md` private); work pushed on branch `web-app`. Built `backend/` (FastAPI: scrypt password, signed cookie, login throttle, private-photo gate via Caddy forward_auth, SQLite sync with history merge; 9 tests), web login + local-first sync (e2e tested phone/laptop), `deploy/` kit. Local backend port 8788 (8000/8765 used by other projects).

**2026-10-27 public demo:** `npm run build:demo` (VITE_DEMO=1): no login/sync, daily-refreshed sample history (tests cover Mon-Sun), "Sample data" labels, neutral quotes, 10 licensed Unsplash/Pexels photos with credits in Settings (club-branded hero rejected). `deploy/deploy.sh` serves it on a 2nd name (`DEMO_DOMAIN`) via Caddy with no auth; skips HTTPS until the name points at the server. Personal Ronaldo/Pinterest photos stay behind the login only.

**2026-10-27 to 11-07:** demo live at sport-coach-demo.duckdns.org (on Peter's portfolio site); private site loads but the login server was never started (password not set), so it runs local-only on the phone. First real field session 2026-11-07 (weak foot session, dribbling added by hand): **voice stops when the PWA is backgrounded or the phone is locked** (iOS suspends it; fix = D32); Peter wants to build sessions from a categorised library (D30).

**2026-11-08:** v3 direction agreed (section 1, D24–D33). `web-app` merged into `main`. Next: phase 1 (Peter sets up Supabase and Cloudflare; Claude builds the GitHub Actions pipeline), then phase 2.

## 9. What Peter should be able to explain (interview notes)

- **RAG:** exercise/drill text → embeddings → vector search by meaning → retrieved items put in the prompt so the model is grounded in real data instead of guessing.
- **Why filter after retrieval:** semantic search can't enforce rules like "nothing that loads the knee badly"; Python filters do. Vector store couldn't filter list-type metadata reliably.
- **Embedding consistency:** the same embedding model must be used at ingest and query time.
- **RAG vs agent:** RAG = stored knowledge; the agent adds tools (web search, data read/write) and decides which to use. Football research is the web-search part.
- **LLM routing:** Gemini primary, Groq fallback — resilience to rate limits and outages.
- **Privacy design:** data owned locally, minimal context sent out.
- **Honest note:** the vector store is LlamaIndex's default in-memory/persisted store; the `chroma_db` folder name is a leftover, not actual Chroma.

## 10. Open questions for Peter

Answered: load rules (1/2/3 games), physio (not now), gym membership (keep for now), web search (use LLM API, free), design direction (black/white/red, Ronaldo), friend drills, cones, hosting, schedule, profile.

1. **Baseline numbers (optional):** 20 m sprint time (or "feels slow/fast"); tiredness at the end of a 90-min casual game (1–10); weak-foot shots on target out of 10; juggles in a row on each foot. Gives the agent a starting point.
2. **Try the session player** on the field and tell me what felt wrong (cue frequency, voice clarity, rest lengths, announcements too chatty).
3. **Review the draft sessions** in `web/src/data/sessions.json` (or in the app): anything you would never do, or want more of?
4. **API keys:** when we reach the agent, you will need to create a free Gemini key and a free Groq key yourself (never paste them in chat; they go in a `.env` file).
