import { useState, useEffect } from 'react'
import axios from 'axios'
import AuthPage from './components/AuthPage'
import HeroSection from './components/HeroSection'
import PredictionForm from './components/PredictionForm'
import ResultCard from './components/ResultCard'
import FeatureImportance from './components/FeatureImportance'
import BatchUpload from './components/BatchUpload'
import './index.css'

// Setup axios interceptor for auth + production URL rewriting
axios.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  // In production (no Vite proxy), strip the /api prefix
  if (import.meta.env.PROD && config.url?.startsWith('/api/')) {
    config.url = config.url.replace('/api/', '/')
  }
  return config
})

export default function App() {
  const [user, setUser] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [apiOk, setApiOk] = useState(null)
  const [checkingAuth, setCheckingAuth] = useState(true)

  // Check saved session on mount
  useEffect(() => {
    const savedToken = localStorage.getItem('token')
    const savedUser = localStorage.getItem('user')
    if (savedToken && savedUser) {
      try {
        setUser(JSON.parse(savedUser))
      } catch { /* ignore */ }
    }
    setCheckingAuth(false)

    axios.get('/api/health')
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false))
  }, [])

  const handleLogin = (userData, token) => {
    setUser(userData)
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setUser(null)
    setResult(null)
    setError(null)
  }

  // Show loading while checking auth
  if (checkingAuth) {
    return (
      <div className="app-wrapper" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="spinner" style={{ width: 32, height: 32 }} />
      </div>
    )
  }

  // Not logged in → show auth page
  if (!user) {
    return <AuthPage onLogin={handleLogin} />
  }

  // Logged in → show role-based dashboard
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
          <div className="header-right">
            <div className="header-badge">
              <span className="status-dot" style={{ background: apiOk === false ? '#ef4444' : '#22c55e' }} />
              {apiOk === null ? 'Connecting…' : apiOk ? 'API Online' : 'API Offline'}
            </div>
            <div className="user-badge">
              <span className="user-role-tag">{user.role === 'teacher' ? '👩‍🏫' : '🎒'} {user.role}</span>
              <span className="user-name">{user.name}</span>
              <button className="logout-btn" onClick={handleLogout}>Logout</button>
            </div>
          </div>
        </div>
      </header>

      {/* ── Hero (shared) ── */}
      <HeroSection role={user.role} />

      {/* ── Role-based Dashboard ── */}
      <div className="main-content">
        {user.role === 'student' ? (
          <>
            {/* Student: Prediction Form + Results */}
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
          </>
        ) : (
          <>
            {/* Teacher: Batch Upload */}
            <div style={{ gridColumn: '1 / -1' }}>
              <BatchUpload />
            </div>
          </>
        )}

        {/* Feature Importance — shared */}
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
