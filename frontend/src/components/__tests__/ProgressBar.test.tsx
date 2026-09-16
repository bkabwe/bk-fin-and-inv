import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import ProgressBar from '../ProgressBar'

describe('ProgressBar', () => {
  it('shows the screened/total counts and percentage', () => {
    render(<ProgressBar screened={250} total={500} currentTicker="AAPL" status="running" />)

    expect(screen.getByText('250 / 500 stocks screened (50.0%)')).toBeInTheDocument()
    expect(screen.getByText('Currently analyzing: AAPL')).toBeInTheDocument()
  })

  it('caps the percentage at 100% when screened exceeds total', () => {
    render(<ProgressBar screened={600} total={500} currentTicker={null} status="running" />)

    expect(screen.getByText('600 / 500 stocks screened (100.0%)')).toBeInTheDocument()
  })

  it('shows a stopped message when status is stopped', () => {
    render(<ProgressBar screened={100} total={500} currentTicker={null} status="stopped" />)

    expect(screen.getByText('⏹ Stopped at 100 stocks')).toBeInTheDocument()
  })

  it('shows a completion message when status is complete', () => {
    render(<ProgressBar screened={500} total={500} currentTicker={null} status="complete" />)

    expect(screen.getByText('✅ Scan complete — 500 stocks screened')).toBeInTheDocument()
  })
})
