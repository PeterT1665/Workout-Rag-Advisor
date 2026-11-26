import { expect, test } from 'vitest'
import { inAppBrowser } from './install'

test('the browser inside TikTok, Instagram and the like is recognised; Safari and Chrome are not', () => {
  const safari = 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1'
  expect(inAppBrowser(safari)).toBeNull()
  expect(inAppBrowser('Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/130.0 Mobile Safari/537.36')).toBeNull()
  expect(inAppBrowser(safari.replace('Safari/604.1', 'musical_ly_34.9.0 BytedanceWebview/d8a21c6'))).toBe('TikTok')
  expect(inAppBrowser(safari + ' Instagram 350.0.0')).toBe('Instagram')
  expect(inAppBrowser(safari + ' [FBAN/FBIOS;FBAV/450.0]')).toBe('Facebook')
})
