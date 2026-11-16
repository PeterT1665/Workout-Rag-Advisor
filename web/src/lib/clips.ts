// Audio clips (mindset and end-of-session) were removed from the app. This deletes any clips
// still saved on the device from older versions. Safe to delete this file once everyone has updated.
export function removeOldClips() {
  try { indexedDB.deleteDatabase('training-media') } catch { /* no IndexedDB */ }
}
