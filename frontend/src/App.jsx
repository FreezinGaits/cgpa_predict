import { useState, useEffect } from 'react'
import axios from 'axios'
import PredictionForm from './components/PredictionForm'
import ResultCard from './components/ResultCard'
import FeatureImportance from './components/FeatureImportance'
import './index.css'

export default function App() {
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)
  const [loading, setLoading] = useState(false)
  const [apiOk, setApiOk]     = useState(null)

  useEffect(() => {
    axios.get('/api/health')
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false))
  }, [])

  return (
    <div className="app-wrapper">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">🎓</div>
            <div>
              <div className="logo-text">CGPA Predictor</div>
              <div className="logo-sub">ML · FastAPI · Real College Data</div>
            </div>
          </div>
          <div className="header-badge">
            <span className="status-dot" style={{ background: apiOk === false ? '#ef4444' : '#22c55e' }} />
            {apiOk === null ? 'Connecting…' : apiOk ? 'API Online' : 'API Offline'}
          </div>
        </div>
      </header>

      {/* ── Hero ── */}
      <div className="hero">
        <div className="hero-tag">🏫 Real Student Data · Ensemble ML · Stacking Model</div>
        <h1>Predict Your Semester CGPA<br />with Machine Learning</h1>
        <p>
          Built on real survey data from college students. Fill in your academic profile
          to get an instant GPA prediction powered by a tuned Stacking Ensemble model.
        </p>
        <div className="hero-stats">
          <div className="hero-stat"><div className="hero-stat-val">583</div><div className="hero-stat-label">Real Student Samples</div></div>
          <div className="hero-stat"><div className="hero-stat-val">67.5%</div><div className="hero-stat-label">±0.5 Accuracy</div></div>
          <div className="hero-stat"><div className="hero-stat-val">85.5%</div><div className="hero-stat-label">±1.0 Accuracy</div></div>
          <div className="hero-stat"><div className="hero-stat-val">0.61</div><div className="hero-stat-label">R² Score</div></div>
          <div className="hero-stat"><div className="hero-stat-val">11</div><div className="hero-stat-label">Models Compared</div></div>
        </div>
      </div>

      {/* ── Main Grid ── */}
      <div className="main-content">
        {/* Left — Form */}
        <div>
          <PredictionForm
            onResult={setResult}
            onError={setError}
            isLoading={loading}
            setLoading={setLoading}
          />
          {error && (
            <div className="error-banner">
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Right — Results */}
        <div>
          {result ? (
            <ResultCard result={result} />
          ) : (
            <div className="card" style={{ height: '100%', minHeight: 400 }}>
              <div className="card-title"><span>🎯</span> Prediction Result</div>
              <div className="empty-state">
                <div className="empty-icon">📊</div>
                <p>Fill in the student profile on the left and click <strong>Predict CGPA</strong> to see results here.</p>
              </div>
            </div>
          )}
        </div>

        {/* Full width — Feature Importance */}
        <div style={{ gridColumn: '1 / -1' }}>
          <FeatureImportance />
        </div>
      </div>

      {/* ── Footer ── */}
      <footer style={{ textAlign:'center', padding:'2rem', color:'var(--text3)', fontSize:'0.78rem', borderTop:'1px solid var(--border)', marginTop:'2rem' }}>
        Built with FastAPI · scikit-learn · Stacking Ensemble · React + Recharts &nbsp;|&nbsp;
        Real data from college student survey &nbsp;|&nbsp; For academic demonstration
      </footer>
    </div>
  )
}
