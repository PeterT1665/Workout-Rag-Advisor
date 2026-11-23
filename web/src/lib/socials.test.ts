import { expect, test } from 'vitest'
import { cleanSocial, socialLink } from './socials'

test('handles: @, pasted profile links and spaces are fine; junk is refused', () => {
  expect(cleanSocial('instagram', '@peter.t_16')).toBe('peter.t_16')
  expect(cleanSocial('instagram', 'https://www.instagram.com/peter.t_16/?hl=en')).toBe('peter.t_16')
  expect(cleanSocial('tiktok', 'https://www.tiktok.com/@peter16')).toBe('peter16')
  expect(cleanSocial('snapchat', 'https://www.snapchat.com/add/pete')).toBe('pete')
  expect(cleanSocial('instagram', 'not a handle!')).toBeNull()
  expect(cleanSocial('instagram', '  ')).toBe('') // empty: removed
})
test('phones: international, spaces and dashes allowed; the link is the chat', () => {
  expect(cleanSocial('whatsapp', '+61 412 345 678')).toBe('+61412345678')
  expect(cleanSocial('zalo', '84-912-345-678')).toBe('+84912345678')
  expect(cleanSocial('whatsapp', '12')).toBeNull()
  expect(socialLink('whatsapp', '+61412345678')).toBe('https://wa.me/61412345678')
  expect(socialLink('instagram', 'pete')).toBe('https://ig.me/m/pete')
})
