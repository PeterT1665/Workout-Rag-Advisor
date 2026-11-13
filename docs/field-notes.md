# Field notes (Peter testing the app on real sessions)

Problems Peter hits while planning and training, and what was done about each.

## 2026-11-13 (before the session)

1. **Drill cards open the whole session.** Tapping a drill card on Home should open that drill's details.
   - Now: drill sheet with photo, sets/time/rest, how to do it, kit. *(v3, goes to staging after the session)*
   - Later: video instructions per drill.
2. **The week plan can't be changed.** E.g. turn today's football into gym, or put a session I customised onto today.
   - Now: Week screen has **Change** on every day (any of your sessions, a recommended one, rest, or back to the plan); session details have **Do it today** and **Pick a day**; Home and Kick off follow the change. Syncs as `weekPlan` once `supabase/migrations/0003` is run. *(v3, goes to staging after the session)*
   - Hybrid days already possible by building a session that mixes gym and football exercises, e.g. MD+2 light lower body (explosive, squat, single-leg RDL) then tight-space control on the field.
   - Open: how a manual change should interact with the matchday rules (warn? adjust the rest of the week?). To discuss.
