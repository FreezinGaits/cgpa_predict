import { useState } from 'react'
import axios from 'axios'

export default function AuthPage({ onLogin }) {
  const [isSignup, setIsSignup] = useState(false)
  const [role, setRole] = useState('student')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const endpoint = isSignup ? '/api/signup' : '/api/login'
      const payload = isSignup
        ? { name, email, password, role }
        : { email, password }
      const res = await axios.post(endpoint, payload)
      localStorage.setItem('token', res.data.token)
      localStorage.setItem('user', JSON.stringify(res.data.user))
      onLogin(res.data.user, res.data.token)
    } catch (err) {
      setError(err.response?.data?.detail || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-bg" />
      <div className="auth-container">
        {/* Left — Branding */}
        <div className="auth-brand">
          <div className="auth-brand-icon">🎓</div>
          <h1>CGPA Predictor</h1>
          <p>Multi-Modal Machine Learning System</p>
          <div className="auth-brand-features">
            <div className="auth-brand-feature">
              <span>🤖</span> Whisper AI · Audio Analysis
            </div>
            <div className="auth-brand-feature">
              <span>👁️</span> Computer Vision · Handwriting
            </div>
            <div className="auth-brand-feature">
              <span>📊</span> Stacking Ensemble · 94.3% Accuracy
            </div>
            <div className="auth-brand-feature">
              <span>🧠</span> 961 Real Students · 20 Features
            </div>
          </div>
        </div>

        {/* Right — Form */}
        <div className="auth-form-panel">
          <div className="auth-form-header">
            <h2>{isSignup ? 'Create Account' : 'Welcome Back'}</h2>
            <p>{isSignup ? 'Join the CGPA Prediction System' : 'Login to your dashboard'}</p>
          </div>

          {/* Role Selector */}
          <div className="role-selector">
            <button
              type="button"
              className={`role-btn ${role === 'student' ? 'role-btn-active role-btn-student' : ''}`}
              onClick={() => setRole('student')}
            >
              <span>🎒</span> Student
            </button>
            <button
              type="button"
              className={`role-btn ${role === 'teacher' ? 'role-btn-active role-btn-teacher' : ''}`}
              onClick={() => setRole('teacher')}
            >
              <span>👩‍🏫</span> Teacher
            </button>
          </div>

          <form onSubmit={handleSubmit} className="auth-form">
            {isSignup && (
              <div className="auth-field">
                <label>Full Name</label>
                <input
                  type="text"
                  placeholder="Enter your full name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>
            )}
            <div className="auth-field">
              <label>Email</label>
              <input
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="auth-field">
              <label>Password</label>
              <input
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={4}
              />
            </div>

            {error && <div className="auth-error">{error}</div>}

            <button type="submit" className="auth-submit" disabled={loading}>
              {loading ? (
                <><div className="spinner" style={{ width: 16, height: 16 }} /> {isSignup ? 'Creating...' : 'Logging in...'}</>
              ) : (
                <>{isSignup ? '✨ Create Account' : '🚀 Login'}</>
              )}
            </button>
          </form>

          <div className="auth-switch">
            {isSignup ? 'Already have an account?' : "Don't have an account?"}
            <button type="button" onClick={() => { setIsSignup(!isSignup); setError(null) }}>
              {isSignup ? 'Login' : 'Sign Up'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
