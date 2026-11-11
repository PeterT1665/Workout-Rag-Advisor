import { render, screen } from '@testing-library/react'
import { expect, test } from 'vitest'
import { PlayerCard, SAMPLE_STATS, overall } from './PlayerCard'
import { SAMPLE_PROFILE } from '../lib/profile'

test('rated card shows the overall and every stat; unrated shows dashes', () => {
  const { rerender } = render(<PlayerCard p={SAMPLE_PROFILE} stats={SAMPLE_STATS} />)
  expect(screen.getByRole('img', { name: `Sam card, overall ${overall(SAMPLE_STATS)}` })).toBeTruthy()
  expect(screen.getByText('86')).toBeTruthy()
  rerender(<PlayerCard p={{ ...SAMPLE_PROFILE, sample: false }} stats={null} />)
  expect(screen.getByRole('img', { name: 'Sam card, unrated' })).toBeTruthy()
  expect(screen.getAllByText('--').length).toBeGreaterThanOrEqual(7)
})
