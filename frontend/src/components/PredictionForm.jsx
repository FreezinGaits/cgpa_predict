import { useState } from 'react'
import axios from 'axios'

const FIELDS = [
  { key: 'midterm',         label: 'Midterm Score',          type: 'range',  min: 0,  max: 100, step: 1,   unit: '/100',  hint: 'Average midterm exam score' },
  { key: 'assignment',      label: 'Assignment Score',        type: 'range',  min: 0,  max: 100, step: 1,   unit: '/100',  hint: 'Average assignment/project score' },
  { key: 'attendance',      label: 'Attendance',              type: 'range',  min: 0,  max: 100, step: 1,   unit: '%',     hint: 'Class attendance percentage' },
  { key: 'study_hours',     label: 'Study Hours/Day',         type: 'range',  min: 0,  max: 12,  step: 0.5, unit: ' hrs',  hint: 'Average daily study time' },
  { key: 'twelfth_pct',     label: '12th Grade %',            type: 'range',  min: 30, max: 100, step: 1,   unit: '%',     hint: '12th board exam percentage' },
  { key: 'tenth_pct',       label: '10th Grade %',            type: 'range',  min: 30, max: 100, step: 1,   unit: '%',     hint: '10th board exam percentage' },
  { key: 'backlogs',        label: 'No. of Backlogs',         type: 'number', min: 0,  max: 20,  step: 1,               hint: 'Failed / pending subjects' },
  { key: 'stress',          label: 'Mental Stress (0–10)',    type: 'range',  min: 0,  max: 10,  step: 1,   unit: '/10',   hint: 'Your self-rated stress level' },
  { key: 'distance',        label: 'Distance from Campus',    type: 'number', min: 0,  max: 200, step: 0.5,             hint: 'Commute distance in km' },
  { key: 'complexity',      label: 'Content Complexity',      type: 'select', options: [['1','Easy'],['2','Medium'],['3','Hard']] },
  { key: 'teacher_feedback',label: "Teacher's Feedback",      type: 'select', options: [['1','Poor / Needs Work'],['2','Average'],['3','Good']] },
  { key: 'participation',   label: 'Discussion Participation', type: 'select', options: [['1','Less Active'],['2','Good Listener'],['3','Shares Stats'],['4','Moderator']] },
  { key: 'prev_prev_gpa',   label: 'Historical GPA (optional)',type: 'number', min: 0, max: 10, step: 0.01, hint: '⭐ Enter GPA from semester before last. Leave 0 if first semester or unknown.' },
]

const DEFAULTS = {
  midterm: 40, assignment: 17, attendance: 80, study_hours: 3,
  twelfth_pct: 80, tenth_pct: 80, backlogs: 0, stress: 3,
  distance: 15, complexity: '2', teacher_feedback: '2', participation: '2',
  prev_prev_gpa: 0, intro_grade: 5, hw_grade: 5,
}

export default function PredictionForm({ onResult, onError, isLoading, setLoading }) {
  const [values, setValues] = useState(DEFAULTS)
  const [introUploading, setIntroUploading] = useState(false)
  const [hwUploading, setHwUploading] = useState(false)
  const [introDetails, setIntroDetails] = useState(null)
  const [hwDetails, setHwDetails] = useState(null)

  const handleChange = (key, val) => setValues(v => ({ ...v, [key]: val }))

  const handleIntroUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setIntroUploading(true)
    setIntroDetails(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await axios.post('/api/grade-intro', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      handleChange('intro_grade', res.data.grade)
      setIntroDetails(res.data.details)
    } catch (err) {
      const msg = err.response?.data?.detail || 'Audio grading failed'
      onError(msg)
    } finally {
      setIntroUploading(false)
    }
  }

  const handleHwUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setHwUploading(true)
    setHwDetails(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await axios.post('/api/grade-handwriting', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      handleChange('hw_grade', res.data.grade)
      setHwDetails(res.data.details)
    } catch (err) {
      const msg = err.response?.data?.detail || 'Image grading failed'
      onError(msg)
    } finally {
      setHwUploading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    onError(null)
    try {
      const payload = {
        ...values,
        midterm:         parseFloat(values.midterm),
        assignment:      parseFloat(values.assignment),
        twelfth_pct:     parseFloat(values.twelfth_pct),
        tenth_pct:       parseFloat(values.tenth_pct),
        study_hours:     parseFloat(values.study_hours),
        attendance:      parseFloat(values.attendance),
        backlogs:        parseInt(values.backlogs, 10),
        stress:          parseFloat(values.stress),
        distance:        parseFloat(values.distance),
        complexity:      parseInt(values.complexity, 10),
        teacher_feedback:parseInt(values.teacher_feedback, 10),
        participation:   parseInt(values.participation, 10),
        prev_prev_gpa:   parseFloat(values.prev_prev_gpa) > 0 ? parseFloat(values.prev_prev_gpa) : null,
        intro_grade:     parseInt(values.intro_grade, 10),
        hw_grade:        parseInt(values.hw_grade, 10),
      }
      const res = await axios.post('/api/predict', payload)
      onResult(res.data)
    } catch (err) {
      const msg = err.response?.data?.detail || 'Prediction failed. Is the API server running?'
      onError(msg)
    } finally {
      setLoading(false)
    }
  }

  const rangePct = (val, min, max) => `${((val - min) / (max - min)) * 100}%`

  return (
    <form onSubmit={handleSubmit}>
      <div className="card">
        <div className="card-title"><span>📋</span> Student Profile</div>
        <div className="form-grid">
          {FIELDS.map(f => {
            const val = values[f.key]
            return (
              <div className="form-group" key={f.key}>
                {f.type === 'range' ? (
                  <>
                    <label className="form-label">
                      {f.label}
                      <span className="form-label-val">{val}{f.unit || ''}</span>
                    </label>
                    <input
                      type="range"
                      min={f.min} max={f.max} step={f.step}
                      value={val}
                      style={{ '--range-pct': rangePct(val, f.min, f.max) }}
                      onChange={e => handleChange(f.key, parseFloat(e.target.value))}
                    />
                    {f.hint && <span className="form-hint">{f.hint}</span>}
                  </>
                ) : f.type === 'select' ? (
                  <>
                    <label className="form-label">{f.label}</label>
                    <select
                      className="form-input"
                      value={val}
                      onChange={e => handleChange(f.key, e.target.value)}
                    >
                      {f.options.map(([v, lbl]) => (
                        <option key={v} value={v}>{lbl}</option>
                      ))}
                    </select>
                  </>
                ) : (
                  <>
                    <label className="form-label">{f.label}</label>
                    <input
                      type="number"
                      className="form-input"
                      min={f.min} max={f.max} step={f.step}
                      value={val}
                      placeholder={f.key === 'prev_prev_gpa' ? '0 = unknown' : ''}
                      onChange={e => handleChange(f.key, e.target.value)}
                    />
                    {f.hint && <span className="form-hint">{f.hint}</span>}
                  </>
                )}
              </div>
            )
          })}

          {/* ── AI Grading Section ── */}
          <div className="ai-section">
            <div className="ai-section-title">🤖 AI-Powered Grading</div>
            <p className="ai-section-desc">Upload files for automatic grading, or adjust the sliders manually.</p>

            {/* Introduction Grade */}
            <div className="ai-grade-row">
              <div className="ai-grade-left">
                <label className="form-label">
                  🎙️ Introduction Grade
                  <span className="form-label-val">{values.intro_grade}/10</span>
                </label>
                <input
                  type="range"
                  min={1} max={10} step={1}
                  value={values.intro_grade}
                  style={{ '--range-pct': rangePct(values.intro_grade, 1, 10) }}
                  onChange={e => { handleChange('intro_grade', parseFloat(e.target.value)); setIntroDetails(null) }}
                />
              </div>
              <div className="ai-grade-right">
                <label className="upload-btn" style={introUploading ? { opacity: 0.5, pointerEvents: 'none' } : {}}>
                  {introUploading ? (
                    <><div className="spinner" style={{ width: 14, height: 14 }} /> Analyzing…</>
                  ) : (
                    <>📤 Upload Audio</>
                  )}
                  <input
                    type="file"
                    accept=".mp3,.wav,.m4a,.webm,.ogg"
                    style={{ display: 'none' }}
                    onChange={handleIntroUpload}
                  />
                </label>
              </div>
            </div>
            {introDetails && (
              <div className="ai-result-box">
                <div className="ai-result-label">✅ Whisper AI Transcript:</div>
                <div className="ai-result-text">"{introDetails.transcript}"</div>
                <div className="ai-result-meta">
                  Words: {introDetails.word_count} · Sentences: {introDetails.sentence_count} · Vocab Richness: {introDetails.vocab_richness}
                </div>
              </div>
            )}

            {/* Handwriting Grade */}
            <div className="ai-grade-row">
              <div className="ai-grade-left">
                <label className="form-label">
                  ✍️ Handwriting Grade
                  <span className="form-label-val">{values.hw_grade}/10</span>
                </label>
                <input
                  type="range"
                  min={1} max={10} step={1}
                  value={values.hw_grade}
                  style={{ '--range-pct': rangePct(values.hw_grade, 1, 10) }}
                  onChange={e => { handleChange('hw_grade', parseFloat(e.target.value)); setHwDetails(null) }}
                />
              </div>
              <div className="ai-grade-right">
                <label className="upload-btn" style={hwUploading ? { opacity: 0.5, pointerEvents: 'none' } : {}}>
                  {hwUploading ? (
                    <><div className="spinner" style={{ width: 14, height: 14 }} /> Analyzing…</>
                  ) : (
                    <>📤 Upload Image</>
                  )}
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png,.bmp,.webp"
                    style={{ display: 'none' }}
                    onChange={handleHwUpload}
                  />
                </label>
              </div>
            </div>
            {hwDetails && (
              <div className="ai-result-box">
                <div className="ai-result-label">✅ Vision Analysis:</div>
                <div className="ai-result-meta">
                  Content Density: {(hwDetails.content_density * 100).toFixed(1)}% · Contrast: {hwDetails.contrast} · Edge Density: {(hwDetails.edge_density * 100).toFixed(1)}% · Line Coverage: {(hwDetails.line_regularity * 100).toFixed(0)}%
                </div>
              </div>
            )}
          </div>

          <button type="submit" className="submit-btn" disabled={isLoading}>
            {isLoading ? (
              <><div className="spinner" /> Predicting…</>
            ) : (
              <> ✨ Predict CGPA</>
            )}
          </button>
        </div>
      </div>
    </form>
  )
}
