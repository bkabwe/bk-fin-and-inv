import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import ScoreCard from '../ScoreCard'

describe('ScoreCard', () => {
  it('renders the numeric score', () => {
    render(<ScoreCard score={72} />)

    expect(screen.getByText('72')).toBeInTheDocument()
  })

  it.each([
    [90, '🟢 STRONG BUY'],
    [70, '🔵 BUY'],
    [55, '🟡 TAKE SMALL POSITION'],
    [40, '🟠 MONITOR'],
    [25, '🔴 DO NOT BUY'],
    [5, '⛔ AVOID']
  ])('labels a score of %i as %s', (score, label) => {
    render(<ScoreCard score={score} />)

    expect(screen.getByText(label)).toBeInTheDocument()
  })
})
