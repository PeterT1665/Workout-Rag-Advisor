# Training Agent: web app (React PWA)

Vite + React + TypeScript + Tailwind v4, installable on iPhone as a PWA.

## Run
```
cd web
npm install
npm run dev -- --port 8080      # http://localhost:8080, and http://<laptop-ip>:8080 on the phone (same Wi-Fi)
npm test                         # unit + UI tests (vitest)
npm run build                    # production build in dist/ (with service worker)
```

## iPhone notes
- Keep-screen-awake and install-to-home-screen need HTTPS: works fully only once deployed.
- Sound and speech start only after tapping Start (iOS rule). Keep the screen on; a locked phone can silence speech and clips.

## Structure
- `src/lib/logic.ts`: pure session maths (steps, cue timing, .ics). Tested.
- `src/lib/player.ts`: session player engine (timer, cues, rest clips, SIUUU). No React. Tested.
- `src/lib/clips.ts`: your audio clips in IndexedDB, on the device only.
- `src/lib/audio.ts`: speech, beeps, clip playback.
- `src/views/*`: screens. `src/App.tsx`: navigation.
- `src/data/sessions.json`: draft sessions (to be replaced by the planning agent).
