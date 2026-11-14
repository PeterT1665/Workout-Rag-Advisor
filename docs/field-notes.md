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

## 2026-11-13 (during the session)

3. **Reordering in the session builder is slow** (adding back squat meant tapping "up" many times).
   - Now: drag each exercise by its grip (⋮⋮) to any position; arrow keys on the grip for keyboards. *(staging)*
4. **Pausing means walking back to the phone** (e.g. fetching the ball).
   - Now: AirPods/earbud controls during a session: press = pause/resume, double-press = next drill, triple = previous. Lock screen shows the session too. Settings switch; on by default. Note: like any audio app it pauses other music while a session runs. *(staging, needs testing on a real iPhone)*

## 2026-11-14

5. **Gym sets counted down like drills.** Sets with reps should run at your own pace.
   - Now: a set shows its reps and a tempo cue (slow down and fast up for strength, max intent for explosive lifts). A count-up timer runs and you tap **Set done** (or press the earbuds) to go to the rest. *(staging)*
6. **Supersets and contrast sets.** Pair two exercises: different muscle groups (push + pull), or a heavy lift followed by a light explosive one of the same movement.
   - Now: in the builder, **+ Superset with above** links an exercise to the one before it. The pair alternates A1, A2, then one rest ("Rest after pair"). Session details and the player show A1/A2.
   - New recommended sessions: **Gym: contrast lifts** (back squat + jump squat, RDL + broad jump, bench + plyo push-up) and **Hybrid: light legs + close control** (the MD+2 example). **Gym: upper body + core** now uses push/pull supersets. *(staging)*
7. **Can't see or change the session while it runs** (e.g. the rack is taken).
   - Now: a **Plan** button in the player shows the whole session (done, now, still to come). What's still to come can be dragged into a new order, swapped for another exercise from the same category, or removed. A Resume after a reload keeps the changes. *(staging)*
8. **Reload lost the session; history missing.**
   - Now: a session survives a reload (**Resume** on Home), and **End session** saves what was done. *(staging)*
9. **Welcome screen repeated "make your player" and "create account"; buttons had no press feedback; the install steps were hard to find; the week step was unclear.**
   - Now: one path on the welcome screen. Every button has press and hover feedback. A "You're in" screen shows the save-your-player and install steps. Match days and gym sessions are separate, bold sections. *(staging)*
10. **Logo.** Keep the 7 for now; explore universal designs (not everyone wears 7). Ongoing in Figma.
11. **Later:** video instructions per drill; how a manual week change should work around match day (to discuss).

## 2026-11-15

12. **Streak on Home** instead of the app name (top left): a flame with the number of days in a row you've kept to the week plan. A day counts if you trained, or if the plan had no training (rest day, game day, a day changed to rest). Today only counts once it's done, and the flame glows then. Tap it to open the Player tab and the training log. *(staging)*
13. **Week plan Edit button** next to "The plan": tap Edit, then tap any day to change it. Outside Edit, session names open the session. *(staging)*
14. **Home workouts:** a new Home library (bodyweight, sofa, table, towel, wall, small-space ball work). Four sessions: legs + power, upper body + core, short full body, and ball in a small space. Every gym session shows **No gym today? Do the home version**, and the mid-session Swap offers home exercises from the same category. *(staging)*
