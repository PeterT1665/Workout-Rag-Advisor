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

## 18/11 (overnight build, on staging)

39. **Team said "create an account" to a signed-in player.**
    - Now: the Team tab also offers **Sign in**, and a weak signal while the app starts no longer drops a signed-in player to guest mode (the saved login is kept and retried).
40. **Languages: English, Tiếng Việt, Español, Português, Français, Bahasa Indonesia.**
    - Every screen, the exercise library (instructions, equipment, session titles), the voice and the notifications. Exercise names stay English.
    - Follows the phone's language on first open; change it in Settings → Language or on the welcome screen.
    - Safety net: `src/i18n/i18n.test.ts` fails if any line is missing in any language or a screen shows text that skips translation. New lines: `node scripts/i18n-catalog.mjs` lists them.
    - Checked on a virtual iPhone (375 and 390 wide) in every language: no screen scrolls sideways, long words wrap properly.
41. **Add friends by QR.**
    - Team → **Show my QR**. A friend scans it: "Peter #7 invited you to be friends" → create an account or sign in → you're friends straight away (no request to accept).
    - Needs migration 0008 (run it before testing).

## 18/11 (day, on staging)

42. **New players open a brand-new app with no instructions.**
    - Now: a first-run tour. The screen dims, one button at a time is lit, and a small box with an arrow says what it does: streak, Player, Week, Kick off, Team, Settings. Next / Skip tour, in every language.
    - Shows for every new or sample player (even on a phone that used the app before). Replay: Settings → **Show the app tour**. It replaces the old "How it works" card.
43. **Changing the language jumped to Home like a glitch.**
    - Now: a quick fade, and the app comes back on Settings, at the same spot, in the new language.
44. **Link previews were empty.**
    - Now: sharing the link shows the laptop page (two phones and the QR) with the app name and a one-line description. Staging links use staging's own image.
45. **Notifications still not arriving.**
    - Now: every app open checks this phone is still registered for notifications at its current address and re-registers it if not (phones renew the address, and the server drops one Apple or Google says is gone). A rest reminder goes to the phone's live address, not an old saved one.
46. **A 6-minute session went in the log like a real one.**
    - Now: a session under 20 minutes never counts toward the streak, not even added to another short one. It stays in the log, greyed with "Under 20 min: doesn't count", and the full-time screen says so.
    - Skipping through a session no longer logs the skipped time: the log saves time actually trained (pauses and skipped drills don't count).

## 18/11 (evening, on staging)

47. **Match day: review every game, and the app rates it.**
    - Game day, 9 am: a notification and a Home box ask **when you play** (kick-off and full time, prefilled from your last game).
    - 15 minutes after full time (9 pm if no times): a notification, a **Match review** box on Home, and the centre ball turns into the review button. It stays open until the end of the next day.
    - The review: score, minutes, position, goals, assists, dribbles completed, key passes, 1v1s won, first touch /10, how tiring /10, your own rating /10, key moments.
    - The app's **match rating** (3.0-10.0, coloured like sports apps): starts at 6.0, goes up with what you did for your position, short games count more per minute, the result and first touch move it a little. A goal and an assist in a win ≈ 7.7; a busy game around them ≈ 8.7.
    - Your last 5 games nudge the card's SHO, PAS, DRI, DEF and PHY by up to 3 points (pace stays from the rating session).
    - **Winning streak** (a draw ends it) and **unbeaten** streak (a draw keeps it).
    - Settings → Notifications has its own **Game days** switch.
    - Needs migration 0009 and send-push redeployed.
48. **Player tab → Performance.** A Player / Matches switch at the top: Player is the card, this week and the training log; Matches has the streaks, average rating and every match report.
    - Tab labels now shrink to fit their space, so long words ("Performance", "Rendimiento", "Trang chủ") fit on every current iPhone in every language.

## 21/11 session (explosive lower body, gym)

49. **Exercises Peter did that the app doesn't have yet.** Three contrast supersets (heavy-ish lift, then the explosive version of the same movement). Peter is sending clips; names below are working names until confirmed.
    - **Superset 1**
      - **A. Split-stance quarter squat (barbell):** staggered stance, not parallel; quarter depth, drive up fast.
      - **B. Split-stance step-up jump:** front foot on a box or platform, back foot on the ground, same stance as A; jump from it as explosively as possible.
    - **Superset 2**
      - **A. Single-leg RDL to knee drive (light barbell, "RDL clean"):** stand on one leg, hinge into a single-leg RDL; on the way up drive the free knee forward and up, like the pull of a clean.
      - **B. Single-leg RDL to box jump (bodyweight):** left foot on the ground, right leg does the bodyweight RDL; on the way up snap the right knee forward and up, then jump onto a box off the left foot.
    - **Superset 3**
      - **A. Landmine curtsy / diagonal reverse lunge:** a lunge stepping back on a diagonal, holding the landmine. Name to confirm from the clip.
      - **B. Skater snap to box jump:** land a skater on the left foot, snap across to the right foot, snap back to the start, then a quick reactive jump onto the box.
    - **What Peter's clips show (22/11):**
      - **RDL clean:** clear. Light barbell held at the hips, single-leg hinge, then a knee drive on the way up.
      - **Landmine lunge:** the end of the bar is held at the shoulder with both hands; he steps back on a diagonal and leans into the bar.
      - **Split-stance jump:** the rear foot looks raised on the box, with the front foot on the floor. That doesn't match the description above; confirm with Peter.
      - **Skater snap:** clear (bound, land, snap back). The box landing isn't in frame.
      - **Not filmed:** the split-stance quarter squat.
      - **Also seen:** a drop off a 20" box, then jumps over a mini hurdle. Ask whether that belongs to this session.
    - **Peter's answers (22/11):**
      - Split-stance jump: the **front** foot is on the platform.
      - Skater snap: he lands on the box at the end.
      - Split-stance quarter squat: bar on the back, normal quarter-squat depth.
      - Box drop + hurdle jump: add it, as an advanced depth jump.
    - **Now (22/11, v3):**
      - All seven are in the library with a 3D demo each: split-stance quarter squat, split-stance step-up jump, single-leg RDL clean, single-leg RDL to box jump, landmine diagonal reverse lunge, skater snap to box jump, depth jump over hurdle.
      - New session **Gym: explosive lower body**:
        - warm-up, then depth jump over hurdle
        - 3 contrast supersets (quarter squat + step-up jump, RDL clean + RDL box jump, landmine lunge + skater snap)
        - Copenhagen plank, cool-down
      - It joins the leg-day rotation and swaps to the home legs session when there's no gym.
50. **3D demos checked against every exercise (22/11).**
    - Med-ball throws held the ball the whole time.
      - Now the ball leaves the hands: the chest pass hits the wall and comes back, the slam hits the floor, the rotational throw goes side-on into a wall.
    - Also fixed:
      - chair dip (the body now lowers)
      - jockey (side-on and low, stepping back)
      - lateral shuffle (now travels between the cones)
      - pogo hops (higher)
      - foam roll (the body rolls over the roller)
    - The other ~95 moves match their exercises.
51. **The coach remembers what matters (22/11).**
    - The chat stays on the phone. What the coach learns is saved on the player's private, synced profile (never shown to friends):
      - an injury ("I pulled my hamstring"): body part, their words, date
      - "it's better now": marks it healed
      - weigh-ins ("I weigh 73 kg") and height ("I'm 180 cm")
    - With an injury on file:
      - The coach flags exercises in today's session that load that part.
      - It mentions the injury when the chat opens.
      - Player → Body lists it with a **Healed** button.
    - Questions ("how do I avoid a hamstring injury?") and normal soreness are not saved as injuries.
52. **Instruction video style, Peter's decisions (22/11).** The approach: FootMax principles (hook → mistake → fix → key cues), built on the app's own 3D engine, not video files.
    - **In the app:** a short loop of 15–25 s in the exercise sheet. The 45–60 s cut is only for TikTok.
    - **Colours:**
      - Red = mistake and green = fix, nothing else; ✕/✓ icons too for colour-blind players.
      - The ball stays white.
      - The figure wears a shirt in the player's team colour, with neutral grey shoes. Done now.
    - **Headline font:** the app's current display font (not a FootMax look-alike).
    - **TikTok export** (9:16 MP4): later, once the in-app version is proven.
    - **Spoken cues:** the app's offline voice reads the key cues, with a mute button.
    - **First batch:** the 30 most-used exercises in the app's sessions.
53. **Exercises from Peter's gym clips (Downloads → Vid Edit Resources → Workout videos, Jul–Aug).**
    - Added with 3D demos: incline dumbbell press, barbell overhead press, seated cable row, face pull, hanging leg raise.
    - Every exercise in those clips that the app already had matches its 3D demo (push-up, pull-up, bench press, single-arm press, med-ball slam). So do the 19/11 ball-work and home-mobility clips.

54. **Player card flips (22/11).**
    - Tap the card and it turns over (a springy 180°; instant with reduced motion).
    - The back shows the last rating session: every test's result (sprint seconds, cone weave time, shots, passes, 5-10-5 time, broad jump cm), the rating it gave, the change since the session before (▲/▼), and when the next one is due.
    - Tap again to turn back.
55. **Rating session: weekly, with a proper warm-up (22/11).**
    - Recommended every week, same day and same warm-up. The card shows the next date; the button says "Rating session due" when it's time.
    - Before the tests, a guided 12-minute warm-up, one step at a time, each with its 3D demo and a timer:
      1. easy jog + leg swings (3 min)
      2. build-up runs at 50 / 70 / 85% (2 min)
      3. wall passes (2 min)
      4. dribbling through the markers (2 min)
      5. pogo hops + easy broad jumps (1 min)
      6. change of direction at 70%, then two sharp ones (2 min)
    - "Already warm: go to the tests" skips it.
56. **More landmine exercises (22/11).**
    - Added with 3D demos:
      - landmine squat
      - landmine single-leg RDL
      - landmine lateral lunge
      - landmine rotational press (a "punch", like striking a ball)
      - landmine rotation (rainbow)
      - landmine row (T-bar)
    - The landmine's floor pivot now stays put (it used to slide with the hands), and one-hand holds work. The existing landmine push press and diagonal lunge look better too.
    - The TikTok accounts (@effordelite, @sarmitok9) still can't be read: TikTok blocks automated access. Peter is to save clips to a folder instead.
57. **In a session, the exercise description opens its video (22/11).**
    - Tap the how-to text of the current exercise ("▶ Watch how") to open its instructions and 3D demo, the same as Up next and the Video button.
    - The session timer keeps running behind the sheet.
58. **Barbell or dumbbells, chosen at kick-off (22/11).**
    - Any session with barbell or landmine lifts asks on the kick-off screen: "What can you get at the gym today? Barbell / Dumbbells". It remembers the last choice.
    - On Dumbbells, each of those lifts switches to its dumbbell version for that session. Sets, rest and supersets stay as planned.
    - The swaps:
      - squats → goblet squat
      - split-stance quarter squat → dumbbell split squat
      - deadlift / trap bar → dumbbell deadlift; RDL → dumbbell RDL
      - hip thrust → dumbbell hip thrust
      - bench → dumbbell bench; rows → dumbbell row; barbell overhead press → dumbbell overhead press
      - cleans → dumbbell hang clean; snatch → dumbbell snatch
      - jerks / landmine push press → dumbbell push press
      - trap-bar jump → dumbbell jump squat
      - landmine squat / single-leg RDL / lateral lunge / diagonal lunge → goblet squat / single-leg RDL / Cossack / dumbbell curtsy lunge
    - 11 new dumbbell exercises, each with a 3D demo.
    - The Plan sheet's per-exercise Swap still works for one-off changes.
59. **Bug: "Log your sets" after a bench + pull-up superset showed only the bench (22/11).**
    - Cause: exercises added, swapped or paired in the Plan sheet mid-session weren't seen by "Log your sets" or Full Time (they read the original plan), so a superset made mid-session lost its second exercise.
    - Fixed: both now read the plan as it is now.
    - Tested: a superset paired mid-session, and a planned superset with different set counts (bench 4, pull-ups 3).
60. **Dumbbell overhead press demo: elbows tucked (22/11).** They start in front of the body (about 30–40°), not flared to the sides, and press up and slightly in. Same for the dumbbell push press. The camera is a little more side-on so the tuck shows.
61. **Peter's answers and the changes they led to (22/11).**
    - **Pull-ups not logged:** the bench + pull-ups superset was already in the session (bench 4 sets, pull-ups 3), so the log card only came after bench's 4th set. Now each exercise in a superset is asked for right after its own last set (pull-ups after round 3, bench after round 4). Tested.
    - **Sets:** the Plan sheet has − / + on every exercise still to come, and on the one under way (never below the set being done).
    - **New exercises in recommended sessions:**
      - Upper body + core: + face pull, and hanging leg raise in place of the dead bug.
      - Upper body power: landmine rotational press in place of the med-ball slam.
      - Core & injury prevention: + landmine rotation.
      - Single leg: landmine lateral lunge in place of the step-up.
    - **Injuries:** auto-swap at kick-off. Exercises that load an injured part swap to a safe one of the same kind, or are left out. The kick-off card lists every change; change it back in the Plan sheet.
    - **Rotation:** "Gym: explosive lower body" is now first in the leg-day rotation.
    - **TikTok:** Peter will save clips from @effordelite and @sarmitok9 into ~/Downloads/training-clips/efford and /sarmiento.
    - **Overnight decisions:** Peter will review them when he sits down; keep them until then.
    - **Staging:** push only when Peter says.
62. **Pallof press demo (22/11, from Peter's screenshot mid-session).** The band was a red line floating at chest height, not attached to anything or held. Now there's a cable column beside the player, with a cable from the pulley to a handle in both hands; the figure presses straight out against it. Cables are thicker and lighter so they show on a phone (face pull and seated cable row too).
63. **TikTok reference clips (22/11, 8 clips saved by Peter, kept out of the repo).** Seven exercises the app didn't have, each with a 3D demo and translations:
    - **Added:** half-kneeling landmine press, landmine single-arm row, lateral step-up (kettlebell, knee drive), kneeling jump to box jump, wall drive (acceleration march), half-kneeling kettlebell halo, stir the pot (stability ball; new ball prop in the demos).
    - **Placed in sessions (to confirm):**
      - Core & injury prevention: stir the pot in place of the dead bug, still paired with the Pallof press.
      - Upper body power: landmine single-arm row in place of the dumbbell row.
      - Lower body power: kneeling jump to box jump in place of the box jump.
      - Explosive: wall drive just before the 10–20 m sprints.
    - **Already in the app, so not added:** landmine curtsy lunge (= the landmine diagonal reverse lunge), Copenhagen plank, back squat → box jump contrast, med-ball rotational throw, cone change-of-direction drills, dribbling change-of-direction circuit.
    - **Peter's own 4 clips** in the folder are shooting sessions plus the 21/11 gym session already noted (49).
64. **Form check in the 3D demos (22/11, first batch of the mistake → fix instruction videos).** On the demo, tap **Form check**: the right way with the key cue (green ✓), then the most common mistake acted out (red ✕, a red glow on the joint that goes wrong), then the fix (green). It loops, about two reps per step, and the coaching voice says each line (mute button on the demo; the choice is remembered on the phone).
    - **Covered:** 42 moves, so about 50 exercises: squats (back, front, half, bodyweight, goblet, landmine), the hinges (RDL, deadlift, trap bar, dumbbell versions, single-leg RDL), bench (barbell, dumbbell), pull-ups, push-ups, plank, split squats, jumps (box, squat, countermovement, broad, dumbbell), pogo, hip thrust and bridge, overhead and push press, rows, kettlebell swing, dead bug, Pallof press, step-up, Nordic. Side plank and Copenhagen have the cue only; a believable "hips dropping" isn't possible on the figure yet.
    - **The mistake grows with the rep:** a back rounds as you hinge and knees cave as you go down. Nothing changes at lockout.
    - **Still to decide:** the default shirt is red, the same as the mistake colour (A16). The ✕/✓ icons and the glow on the joint keep them apart, but a neutral default shirt would make it clearer. TikTok export comes later.
65. **Peter's answers to the checklist (23/11), and what was built from them.**
    - **Kept as they are:** A1 (3D demo first), A3, A6, A7, A8, A9 (keep the upper body + core changes too), A10, A14, A15, A16 (red shirt), A17. Peter tests the B list on staging.
    - **"Add your own video" is gone**: the 3D demos replace it. The saved links are no longer synced or shown.
    - **Button renamed AI coach** (A5). The coach answers from built-in logic (A2) and now also covers:
      - how to use every part of the app: changing the plan, building or customising a session, mid-session edits, kit, injuries, rating sessions, match reviews, language, earbuds, account, Form check
      - training advice with exercise cards: top speed, acceleration, agility, stamina, jumping, shot power, strength, dribbling, first touch, passing, hamstring/groin/knee-ankle prevention, season planning, match-day food
      - how to play better by position: winger, striker, midfielder, defender, full-back, goalkeeper
      - Advice cards leave out exercises that load an injury on file.
      - Making a brand-new exercise isn't possible yet; the coach says so. **Ask Peter if it should be built.**
    - **Injuries (A11–A13) are a hard filter**, saved in the app's own terms: body part (14 parts; calf and ankle now separate, plus groin, core, chest, arm), tissue (muscle, joint, ligament, tendon, bone) and what happened (strain, tear, sprain, tendon pain, bruise, fracture, pain). The player's own words aren't stored.
      - The coach asks **how bad it is** (Light / Moderate / Serious chips; Body has the same switch).
      - Light: exercises that mainly work the part are swapped out for 2 days, then it clears by itself.
      - Moderate: everything that loads it is out, and for a leg injury no jumps, sprints or Olympic lifts. A leg day it guts becomes the gym session it leaves most of.
      - Serious: kick-off is blocked, with rest and physio advice.
    - **Every exercise lists what it works and what else it loads** (muscles and joints, on the exercise sheet), and that's what the injury filter uses. 40 big lifts and jumps have a "Stay safe" technique note.
    - **The app learns each player's way** (football and gym): swaps, exercises left out or added, sets and supersets changed in a session are remembered when it's finished. Next time that session comes back that way (kick-off lists the changes, with Use the original). The match-day rules still win: no heavy legs or Olympic lifts within 2 days of a game, no extra sets the day before.
    - **Streak button → calendar**: a month view of trained days, kept rest days, missed days and matches. Tap a day to see it and **Share** a Strava-style picture: your photo (it stays on the phone), the session, match or rating results, and your streak. Design is a first version until Peter's inspiration photos arrive.
    - **Testing the welcome intro**: it shows to anyone without an account, so open the app in a private/incognito window (or sign out).
66. **Streak calendar and share pictures from Peter's inspiration (Locket calendar, Strava card), 23/11.** His picks: photo at Full time; gym card shows time + sets, estimated calories and best lift; shots counted at Full time; a see-through sticker as the main export.
    - **Streak flame → a small window** (Locket-style), not a full page. Month of rounded tiles:
      - trained day: the day's photo (or a ball / dumbbell / leaf icon) in a gold frame, with the streak count on the last day of each run
      - rest day kept: a dot
      - game day: a purple-blue flame
      - A switch flips to the **game streak** (unbeaten, wins in a row): game days show W/D/L in purple-blue. Peter is sending inspiration for the win/lose streak design.
      - Tap a day: its photo (add or change it), what was done, and Share for each.
    - **Full time** asks for an optional **photo of today** (it stays on the phone; it's the calendar tile and the share background). Sessions with finishing drills get a **shot tally**: shots, on target, goals (− / +, kept in order). After saving there's a **Share** button.
    - **Share picture, Strava-style**: the title and date, three stats stacked big in the middle, a graphic, then FOOTBALL COACH and the streak. Three looks: see-through **sticker** (previewed on the day's photo), **black**, or **on my photo**.
      - Gym: time · sets, calories (est.), best lift with NEW BEST when it beats every earlier session. Graphic: every set as a bar (height = weight moved).
      - Football: time, on target / shots, goals. Graphic: a shot map on a goal (goals in the net, saves on the line, misses outside). Without a tally: time · drills, calories, effort.
      - Match: result and score, rating, goals · assists, with the purple-blue flame and "N games unbeaten".
      - Rating session: three test results and the six card stats as bars.
    - **Calories are an estimate:** MET × body weight × hours. Gym 3.4–7 MET and football 4.5–9.5 MET depending on the session's effort; recovery 2.5. Always labelled "est.".
67. **Overnight 23/11: game streak design, simpler share card, research batch.**
    - **Game streak (Peter's Valorant reference, locked):** green win, red loss, grey draw. **Purple fire = win streak, blue fire = unbeaten.** The calendar's Games tab shows both counts, W/D/L tiles and a results list: VICTORY/DEFEAT/DRAW in its colour, score, rating, G/A, a Match MVP tag at 8.5+, and the streak flame each game was part of. Tap a game to share it.
    - **Share picture, simpler** (Peter: "just enough for branding", less AI-looking): three stats only (label over a big number), one graphic, and one brand line "🔥 12 FOOTBALL COACH" (the flame's colour follows the streak type). No title, date or tags. Font: Plus Jakarta Sans, a social-app look, loaded only when a picture is drawn. Gym: time, calories (~, an estimate), best lift (label "Bench press · New best", value "80 kg × 5"), set bars. Football: time, on target, goals, shot map.
    - **From research.md** (Peter's research session: @effordelite, @sarmitok9, kc9performance, ozzy.ore, antonypaps, coachc4_, prolific.soccer, trainwithstan, hirtlocker...), 20 exercises with 3D demos, muscles/joints and translations:
      - single-leg lateral rebounds, toe-down wall strikes (ankle lock for shot power), wall passes with number calls
      - Bulgarian split squat jump, broad jump to cut, box toe taps, seated box jump, single-leg box jump
      - Copenhagen short lever, falling starts, get-up sprints, ankling, 5-10-5 shuttle, backpedal to hip turn
      - ab wheel rollout, side plank thread the needle, standing hip CARs, banded hip-flexor march, med ball rotational slam, landmine squat to press
    - **Number calls:** the voice now calls random numbers in the number-call drill (even: pass back first time, odd: turn and go), never the same number twice in a row, shown big on screen. Same timing settings as the left/right cues.
    - **New football session** "Speed: acceleration and agility", in the order the research found coaches use: rhythm (ankling, wall drive) → horizontal plyos (broad jump to cut) → sprint starts (falling, get-up) → agility (5-10-5, backpedal turn). It joins the football rotation.
    - **Half-kneeling landmine press cue** updated from the research: press along the bar's line and lean into it, don't press back.
    - **Fixed: running demos.** Sprint, acceleration starts, bounds and the new speed drills had no mid-stride pose, so the figure knelt between strides. Every run is now a six-phase stride (drive, touchdown, mid-stance, each leg).
68. **Own exercises (Peter 23/11) and the final research copy.**
    - **Add your own exercise:** at the bottom of the exercise list (building a session, or the Plan during one) there's "Can't find it? Add your own exercise": name, what it works (body-part chips), football/gym/home/mobility, an optional how-to.
      - It's usable straight away (category "My exercises", 3 × 10 by default, "3D demo on the way" on its sheet).
      - The injury filter uses the parts picked. It syncs with the account ('myExercises').
      - When signed in it's sent to `exercise_requests` (**migration 0013, Peter runs it**; before that, requests wait on the phone and are sent later). RLS: players add and read their own; max 20 a day; status requested/building/added/declined is set by the team in the dashboard.
    - **The coach keeps saving weight and height** (Peter: the planner and coach need them for goals like gaining weight or sprinting faster).
    - **Final research copy** (`~/Downloads/Claude outputs/research.md`, section B): better cues on 10 existing exercises:
      - back squat: explosive intent, heels raised for depth
      - Bulgarian: front knee at 90°, slight lean
      - RDL: feel it down the back of the legs, explosive up
      - Copenhagen: start with the short lever
      - dead bug and plank: harder versions with a plate
      - landmine rotation: hips lead, then fast
      - hip thrust: knees at 90°, hold at the top
      - step-up: sprinter version
      - single-leg hamstring iso
    - **`onboarding-research.md` (07:20)** has a brief for a new first-run flow: welcome tour, one question per screen, level, goals, gym experience, equipment, injuries, "Building your week", plan reveal. Not built yet: waiting on Peter's answers.
69. **New first-run onboarding (brief from `onboarding-research.md`; Peter 23/11: build now, 8 positions, goals change picks and order, not the week's structure).**
    - **New players** (guest "Get started", or a new account):
      - a 3-screen skippable tour, using the app itself: a week strip around a Wednesday game, a live 3D demo, streak flames
      - one question per screen, with a progress bar and Back:
        - name + number
        - position on a tappable pitch (GK, CB, FB, DM, CM, AM, W, ST; up to 3), with a one-line role card
        - stronger foot
        - level (casual / amateur club / semi-pro / academy-pro)
        - up to 2 goals (faster, stronger, stamina, better on the ball, injury-free)
        - games: tap once for competitive, twice for casual
        - gym sessions a week + gym experience (new / some / experienced)
        - where you train (gym / home / pitch) + kit (ball / wall / cones)
        - anything hurting: area, right now or the last 2 weeks, niggle / limits training / can't train
      - **"Building your week"** (~3 s): one line per answer, e.g. "Game Wednesday, so nothing heavy the day before", "New to the gym: technique first", "Hamstring: exercises that load it are swapped out".
      - **Plan reveal**: the week strip, today's session (already fitted to injuries and kit), the planner's reasons, one button "Let's train", then the existing "You're in".
      - Answers are saved on the phone after every step, so a closed app resumes where it stopped.
      - No birthday, height or weight here (the coach records those later).
    - **Edit player**: the same questions without the tour or injuries (those live in Record → Body), plus colours and role model at the end.
    - **Positions:** 8 roles. Old 'midfielder' and 'defender' read as CM and CB. Match rating and library tags use the role's family (keeper → defender weights). Coach: keepers and full-backs get their own "play better" answer.
    - **`lib/fit`:**
      - Goals order each rotation (sessions that train them come first).
      - New to the gym: no Olympic lifts or advanced jumps (a simpler one of the same kind), 3 sets at most. Casual level: 3 sets at most.
      - No gym: the home version.
      - No wall / cones / ball: those drills swap or drop.
      - Applied to today's session and at kick-off. Injuries still go last.
    - **A sprint day the injury guts now swaps too** (the "safest instead" rule covers football days, not only gym).
    - Onboarding answers are private (profile, never on Team).
70. **Social links (Peter 23/11; in-app chat parked for now).**
    - The Team leaderboard is **friends-only** (RLS 0005: you and accepted friends), so a future chat would be friends-only too.
    - **Your socials** (Team tab, under your username): Instagram, TikTok, Snapchat, Messenger, WhatsApp, Zalo.
      - Handles accept @ or a pasted profile link. Phone numbers need the country code.
      - Bad values are refused, and each saved one has "Test the link".
      - A note warns that friends see phone numbers too.
    - **On a friend's card** (tap them on the leaderboard): "Message on" buttons. Instagram opens the DM (ig.me), WhatsApp and Zalo open the chat (wa.me, zalo.me), Messenger opens m.me.
    - Stored in `profiles.socials` (**migration 0014, Peter runs it**), readable only by you and friends, size-checked. Before the migration, Save says it couldn't save.
    - Chat later: threads of 2+ (friends only), live messages, push, block/report, rate limit, messages auto-deleted after 7–30 days. Peter to decide groups at launch and how long messages are kept.
71. **Staging test fixes (Peter 24/11, from his phone).**
    - **Share picture:** no best-set line, no red set bars, no app name until there's a brand. Time and calories, the streak big, then name and number. A **Crop photo** button (story-shaped crop, for that picture only).
    - **Streak calendar glitch:** it opened offset because the screen-in transform made its wrapper the box for its fixed layer. Calendar and Coach now render outside that wrapper.
    - **Week chips:** Casual / Comp shrink to fit, medium weight not bold.
    - **Cards:** the exercise card picture is a still of its 3D demo (`npm run demo:stills`, public/demo-still, runtime-cached); the drill sheet drops the stock photo when there is a demo.
    - **Kit check:** gym days show training shoes, a med ball (not a football) and water. Photos pending Peter's yes: Unsplash fS3tGOkp0xY (shoes), g6uG1yZ_gq0 (med ball), Kf6UgCx5mb8 (water bottle).
    - **Mid-thigh pull demo** has a power rack with safety pins just above the bar.
    - **Kick off** has a Back button once the card shows.
    - **More menu** replaces the gear and brush on Home: a small window under the button, the page blurred behind it: Make it yours, Exercises, Sessions, Settings. "Make it yours" is gone from Settings. **Exercises** is a new page (by place and category, tap one for the how-to and 3D demo) with Sessions and + Build a session buttons.
    - **Scroll bounce:** removed `overscroll-behavior: none` on html so the page rubber-bands like a normal app (check on the iPhone).
72. **Rating rebuilt from the FIFA research (Peter 24/11: yes to all four).** Source: `~/Downloads/training-clips/rating-research.md` (not in the repo).
    - **Scale:** 75 = an average pro in that quality, 85 = top-flight standout, 95 = among the best. Sprint, jump and shuttle anchors from pro studies (gate-timed); **dribbling, shooting and passing anchors are judgment** (no norms for these tests), to be re-fitted to real users' results (~50–100 rated players).
    - **Re-scored from raw** every time (`rescored`), so saved sessions follow the current anchors. Old sessions keep the full-goal shooting scale (cap 80); new ones use `v: 2`.
    - **Corner shots** replaces shots on target (two 1 m zones inside the posts). The shuttle is named an agility test (still the DEF stat).
    - **Limits:** impossible results are refused (Next disabled); world-class ones need a photo and a friend (shown at the end). A result far from the player's others is flagged too.
    - **Overall is position-weighted** (first position; plain average for a goalkeeper or none), missing tests left out. Gold 75 / silver 65 unchanged.
    - **First session = starting point**; skipped tests stay blank; "Make this my starting point" under the card; **Ask a friend to verify** opens the share sheet (no verified badge yet).
    - Not done: scoring dribbling on the "deficit" (time with the ball minus without): the research gave no anchors for it.
    - Peter's own card: 76 gold → about 67 silver with the new scale (sprint timed by phone is the weakest input; weave 6.19 s is flagged as needs-proof).
73. **How rating works + a card goal (Peter 24/11).**
    - **Record tab:** a black **How rating works** button under New rating session (the idea of a button on the card back was dropped). It opens a short page: the six stats (name big, test small), the FIFA-scale line, a small gold card "75 is an average pro" with silver and bronze below, the overall and how it is weighted for your position. A red **What does my rating mean?** opens a small window; a small **Full table** link shows result-per-rating for every test and the position weights (generated from the scoring anchors, so it can't drift).
    - **The window:** your card (a bit blurred) next to the goal card, a goal stepper (type it, or Silver 65 / Gold 75 / Top 85), and the quickest ways: the stats that count most for your position, the same rise on each, at most 3 stats and +15 each; when that isn't enough it says where the steps get you. Already at the goal: "Aim higher?" with the next tier. Shooting from an old full-goal session is left out ("redo it with the corner test").
    - **Goal:** `profile.goal`, default 75 (gold: everyone works toward gold). Asked in the player survey (new step "Your card goal": Silver 65, **Gold 75**, Top 85) and in Edit player; changeable in the window.
    - **How your week works:** the references ("Learned from", the Ref pills) are removed; the five rules stay. Peter may cut the rest down later.
    - Peter's saved cone weave (6.19 s) was one way only; the real course time is about 11.6 s, which would score about 71. His saved session can't be edited by Claude (read-only access); his next rating session fixes it.
74. **Calories rebuilt (Peter 24/11; research `~/Downloads/training-clips/calorie-research.md`).**
    - Shown as a **range** (e.g. ~130–170): honest error is about ±20%, and a watch reads lifting higher (a note says so in the Record day card).
    - **Summed per exercise** when the session is known (`lib/kcal.ts`): a max sprint, max jump or max isometric hold costs a fixed amount however fresh you felt after; rests count at a low MET; strength and football blocks use a MET set by the session effort; scaled to the minutes you trained. Otherwise whole-session MET by type and effort (gym 3 + 0.3×effort, football 4 + 0.5×effort, home 2.5 + 0.4×effort, recovery 2.3).
    - Peter's activation session (effort 4) is about 110–145 kcal, not 160-300: the research says the old figure wasn't low. Anchors for per-rep costs are judgment (the max isometric hold is the weakest).
    - **Check-in:** the load question stays "How hard was the whole session?" (a light activation day SHOULD count as low load); a line under it says to think of the whole session.
