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
15. **Streak redemption** instead of losing the streak over one miss:
    - Any session keeps a planned day: no pitch, do the gym; no gym, do a home workout; nothing proper, do recovery.
    - A day missed entirely can be made up with an **extra** session of at least 20 minutes within the next 2 days. It must be on top of that day's own plan, or on a rest or game day. Each extra session makes up one missed day.
    - Until the deadline the streak isn't broken, and Home shows **Make up for <day>** with a ~30 min home workout and a recovery session ready. *(staging)*

## 2026-11-15 (after the session)

16. **Music stopped during a session.** The AirPods controls keep a silent track playing, and iPhone gives the audio to the page, which pauses Spotify or Apple Music. Web apps can't share it.
    - Now: AirPods controls are **off by default**, so music keeps playing. Turn them on in Settings if you'd rather pause with the earbuds (it stops music). *(staging)*
17. **A 5-minute session saved the streak.**
    - Now: a training day needs **20+ minutes** of training. Logged sessions can be deleted (✕ in the Player tab's Training log), and a delete stays deleted on every device. *(staging)*
18. **No more mindset.**
    - Now: mindset clips, the end-of-session clip and the mindset photo slots are removed. Clips saved on the phone are deleted on next open. Quote cards stay for now (to confirm). *(staging)*
19. **Chart by kind of session.**
    - Now: "Minutes trained this week", stacked by football, gym, recovery & mobility, and home. It has a legend and tap details. The "N done" count is gone. *(staging)*
20. **Coach → Week, Team tab.**
    - Now: **AI Coach** is a button at the top of the Week page. The tab is now **Team**: a preview with you on the leaderboard and friend search marked "soon". Real friends and leaderboards need the backend redesign (docs/system-design.md). *(staging)*
21. **Video instructions per exercise (priority).**
    - Now: paste a TikTok, YouTube or Instagram link on any exercise (drill sheet, builder, or **Video** during a session). It plays inside the app, and **Find one on TikTok** searches for the exercise.
    - Links are private to you (creators own them) and sync as `videos` once `supabase/migrations/0004_videos.sql` is run.
    - Later: licensed videos in the shared library. *(staging)*
22. **Next week's plan on Sunday.**
    - Now: on Sundays the Week page shows **Next week** under this week, editable the same way. On Monday it's simply this week.
    - The Sunday notification needs push (server job), planned in docs/system-design.md. *(staging)*
23. **Repo clean-up:** removed the retired FastAPI backend and EC2 deploy scripts. CI now runs the web app and the Python week-rule tests only.
24. **Product focus** (Peter):
    - Core: plan the week around the game, correct instructions per exercise, detailed football sessions.
    - Growth: friends leaderboard (streaks), player card rating, training with friends.
25. **Quotes without names.** The app showed your role model's name under every quote, whoever said it (the "Speed didn't say that" moment). Names are gone; quotes stay. *(staging)*
26. **Team, phase 1.** Pick a username and photo, find friends by exact username + shirt number, send and accept requests. The leaderboard ranks friends by streak, then rating, and shows who's online. Tap a friend to see their card. Friends-only. Needs `supabase/migrations/0005_team.sql` to be run. *(staging)*
27. **"Light" sessions were the same session with a label** (upper + core took the same time light or not).
    - Now: planner v4, learned from a pro's week (docs/training-references.md). Each gym day has a job by distance to the game:
      - legs / contrast at MD-4+
      - core & injury prevention at MD-3
      - upper body power at MD-2
      - a ~23 min activation at MD-1
    - Three new sessions. The Week page shows MD-2 etc. under each day, and Home shows the day's job. *(staging)*
28. **Second pro reference** (docs/training-references.md #2): **hard days hard, easy days easy.** The high-intensity football session now shares the MD-4 legs day, and MD-3 stays light (core, no football). *(staging)*
29. **The week explanation was a block of text.**
    - Now: a small **How your week works** button (e.g. "1 game · 2 football · 3 gym · 1 recovery").
    - It opens a sheet: each day with its matchday and job, the planner's rules, and the pro references each rule came from. *(staging)*

## 2026-11-16 (after the gym)

30. **Dragging exercises was buggy.** Supersets didn't move together, and the page scrolled while dragging.
    - Now: a new drag in the builder and the in-session Plan.
      - A superset moves as one block.
      - The page stays still and only scrolls when you hold the exercise at the top or bottom edge.
      - Others slide out of the way.
      - Keyboard arrows still work. *(staging)*
31. **Changing the session at the gym.**
    - Now: Plan during a session has drag, **+ Superset with above** (combine two exercises on the spot), **+ Add exercise** (the full library, including Home), **Swap** (now opens and closes smoothly) and **remove**.
    - Removing the first exercise of a superset leaves its partner as a normal exercise. *(staging)*
32. **No crop for the card photo.**
    - Now: picking a card photo (Player tab → **Change photo**, or Make it yours) opens a crop screen. Drag, pinch or slide to zoom, inside a circle like a profile picture.
    - The Team profile picture uses the same crop. *(staging)*
33. **Rest-over notification** ("Come back to training. Lock in, let's go" / "No more doom scroll, let's roll"). Needs server push on iPhone. Waiting on Peter's go-ahead (see the 16/11 chat).
34. **Rest-over notifications and the Sunday reminder** are built.
    - Leave the app mid-rest and you get called back when the rest ends: "Rest is over · Bench press". Lines: "Come back to training. Lock in, let's go." / "No more doom scroll. Let's roll." / ...
    - Sunday 18:00: "Plan next week", which opens the Week page.
    - Ask: a one-time **Turn on** card on Home; Settings → Notifications has a toggle for each.
    - Needs an account and the app on the Home Screen.
    - Setup steps: docs/push-setup.md. *(staging, waiting on Peter's Supabase setup)*

## 17/11 session

35. **Card photo still square after cropping.**
    - Now: the card shows the photo as a circle with a thin ring, like the crop. *(staging)*
36. **Two installs on the same account both buzzed at the end of a rest.**
    - Cause: a rest call-back went to every device on the account.
    - Now: it goes only to the device whose session it is; "Plan next week" goes to the newest device. Migration 0007. *(staging)*
37. **Notifications only worked sometimes, then stopped.**
    - Cause 1: the call-back was booked as you left the app, and iPhone suspends the app before that request gets out. Now it's booked when the rest starts and dropped if you stay to the end.
    - Cause 2 (likely): every rest notification had the same tag, and iPhone can swap a same-tag notification in silently. Now each has its own tag; old rest notifications are cleared.
    - Settings → Notifications has **Send a test**. send-push logs why a send failed. *(staging)*
38. **Tap Up next during a rest to see the next exercise.**
    - Now: it opens the next exercise's sets, how-to and video. *(staging)*
