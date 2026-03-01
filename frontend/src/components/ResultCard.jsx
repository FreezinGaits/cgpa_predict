import { useEffect, useState } from 'react'
import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from 'recharts'

const GRADE_COLORS = {
  Outstanding:   { bg: 'rgba(34,197,94,0.12)',  border: 'rgba(34,197,94,0.35)',  text: '#22c55e' },
  Excellent:     { bg: 'rgba(6,182,212,0.12)',   border: 'rgba(6,182,212,0.35)',  text: '#06b6d4' },
  'Very Good':   { bg: 'rgba(99,102,241,0.12)',  border: 'rgba(99,102,241,0.35)', text: '#818cf8' },
  Good:          { bg: 'rgba(168,85,247,0.12)',  border: 'rgba(168,85,247,0.35)', text: '#c084fc' },
  Average:       { bg: 'rgba(245,158,11,0.12)',  border: 'rgba(245,158,11,0.35)', text: '#f59e0b' },
  'Below Average':{ bg:'rgba(239,68,68,0.1)',   border: 'rgba(239,68,68,0.3)',   text: '#f87171' },
  'At Risk':     { bg: 'rgba(239,68,68,0.15)',   border: 'rgba(239,68,68,0.4)',   text: '#ef4444' },
}

const INSIGHT_ICONS = ['⚠️','📉','😓','🎯','📚','✅','📍','📈']

export default function ResultCard({ result }) {
  const [animCGPA, setAnimCGPA] = useState(0)

  useEffect(() => {
    let start = 0
    const target = result.predicted_cgpa
    const dur = 800
    const step = 16
    const inc = (target / dur) * step
    const timer = setInterval(() => {
      start += inc
      if (start >= target) { setAnimCGPA(target); clearInterval(timer) }
      else setAnimCGPA(start)
    }, step)
    return () => clearInterval(timer)
  }, [result.predicted_cgpa])

  const gc = GRADE_COLORS[result.grade_band] || GRADE_COLORS['Good']
  const riskPct = (result.predicted_cgpa / 10) * 100
  const radialData = [{ value: result.predicted_cgpa, fill: gc.text }]

  return (
    <div className="result-panel">

      {/* ── CGPA Meter ── */}
      <div className="card meter-card">
        <div className="card-title"><span>🎯</span> Prediction Result</div>
        <p className="meter-label-top">Predicted Semester CGPA</p>

        <div className="meter-wrapper">
          <ResponsiveContainer width={220} height={220}>
            <RadialBarChart
              cx="50%" cy="50%"
              innerRadius="70%" outerRadius="100%"
              startAngle={210} endAngle={-30}
              data={radialData}
              barSize={14}
            >
              <PolarAngleAxis type="number" domain={[0,10]} angleAxisId={0} tick={false} />
              <RadialBar
                background={{ fill: 'rgba(255,255,255,0.05)' }}
                dataKey="value"
                angleAxisId={0}
                cornerRadius={7}
              />
            </RadialBarChart>
          </ResponsiveContainer>
          <div style={{ position:'absolute', inset:0, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', pointerEvents:'none' }}>
            <div className="cgpa-display">{animCGPA.toFixed(2)}</div>
            <div className="cgpa-scale">/ 10.00</div>
          </div>
        </div>

        <div className="confidence-range">
          Confidence Range: {result.lower_bound} – {result.upper_bound}
        </div>

        <div>
          <div className="grade-badge" style={{ background: gc.bg, border: `1px solid ${gc.border}`, color: gc.text }}>
            {result.grade_band} · {result.risk_level} Risk
          </div>
        </div>

        <div className="risk-bar-container">
          <div className="risk-bar-label">
            <span>0 — At Risk</span>
            <span style={{ color: gc.text }}>{result.grade_band}</span>
            <span>10 — Outstanding</span>
          </div>
          <div className="risk-bar-track">
            <div
              className="risk-bar-fill"
              style={{ width: `${riskPct}%`, background: `linear-gradient(90deg, #ef4444, #f59e0b, ${gc.text})` }}
            />
          </div>
        </div>

        <p style={{ marginTop:'1rem', fontSize:'0.8rem', color:'var(--text2)', lineHeight:1.5 }}>
          {result.grade_description}
        </p>

        <p style={{ marginTop:'0.5rem', fontSize:'0.7rem', color:'var(--text3)' }}>
          Model: {result.model_name} · ±1.0 accuracy: 85.5% on test data
        </p>
      </div>

      {/* ── Insights ── */}
      <div className="card">
        <div className="card-title"><span>💡</span> Key Insights</div>
        <div className="insights-list">
          {result.key_insights.map((ins, i) => (
            <div className="insight-item" key={i} style={{ animationDelay: `${i * 0.1}s` }}>
              <span className="insight-icon">{INSIGHT_ICONS[i % INSIGHT_ICONS.length]}</span>
              <span>{ins}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
