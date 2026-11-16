import { describe, expect, test } from 'vitest'
import { cleanLink, embedFor, setVideo, videoFor } from './videos'

describe('exercise videos', () => {
  test('links become embeddable players where the site allows it', () => {
    expect(embedFor('https://www.tiktok.com/@coach/video/7312345678901234567?lang=en')).toEqual({ src: 'https://www.tiktok.com/embed/v2/7312345678901234567', tall: true })
    expect(embedFor('https://youtu.be/abc123')?.src).toBe('https://www.youtube-nocookie.com/embed/abc123')
    expect(embedFor('https://www.youtube.com/shorts/xyz')).toEqual({ src: 'https://www.youtube-nocookie.com/embed/xyz', tall: true })
    expect(embedFor('https://www.instagram.com/reel/C1abc/')?.src).toBe('https://www.instagram.com/reel/C1abc/embed')
    expect(embedFor('https://vm.tiktok.com/ZMabc/')).toBeNull() // short link: opened instead
  })

  test('pasted text is cleaned to a link; one link per exercise name', () => {
    expect(cleanLink('look at this https://www.tiktok.com/@a/video/1 !')).toBe('https://www.tiktok.com/@a/video/1')
    expect(cleanLink('not a link')).toBeNull()
    setVideo('Back squat', 'https://youtu.be/a')
    expect(videoFor('back squat')).toBe('https://youtu.be/a')
    setVideo('Back squat', null)
    expect(videoFor('Back squat')).toBeUndefined()
  })
})
