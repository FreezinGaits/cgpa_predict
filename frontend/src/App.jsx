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
              <div className="logo-sub">ML · Whisper · OpenCV · FastAPI</div>
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
        <div className="hero-tag">🏫 Multi-Modal ML · Audio + Vision + Survey · Stacking Ensemble</div>
        <h1>Predict Your Semester CGPA<br />with Multi-Modal Machine Learning</h1>
        <p>
          Built on real data from 961 college students — combining survey responses,
          audio introductions (Whisper AI), and handwriting analysis (Computer Vision)
          to predict GPA with a tuned Stacking Ensemble model.
        </p>
        <div className="hero-stats">
          <div className="hero-stat"><div className="hero-stat-val">961</div><div className="hero-stat-label">Real Student Samples</div></div>
          <div className="hero-stat"><div className="hero-stat-val">78.8%</div><div className="hero-stat-label">±0.5 Accuracy</div></div>
          <div className="hero-stat"><div className="hero-stat-val">94.3%</div><div className="hero-stat-label">±1.0 Accuracy</div></div>
          <div className="hero-stat"><div className="hero-stat-val">0.763</div><div className="hero-stat-label">R² Score</div></div>
          <div className="hero-stat"><div className="hero-stat-val">20</div><div className="hero-stat-label">Features Used</div></div>
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
        Built with FastAPI · scikit-learn · Whisper AI · OpenCV · Stacking Ensemble · React + Recharts &nbsp;|&nbsp;
        Multi-modal data from 961 college students &nbsp;|&nbsp; For academic demonstration
      </footer>
    </div>
  )
}
