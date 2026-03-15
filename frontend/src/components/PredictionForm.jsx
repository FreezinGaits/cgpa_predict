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
  { key: 'intro_grade',     label: 'Introduction Grade',      type: 'range',  min: 1,  max: 10,  step: 1,   unit: '/10',   hint: '🎙️ Quality of verbal introduction (from Whisper AI transcription analysis)' },
  { key: 'hw_grade',        label: 'Handwriting Grade',       type: 'range',  min: 1,  max: 10,  step: 1,   unit: '/10',   hint: '✍️ Quality of handwritten notes (from Computer Vision image analysis)' },
]

const DEFAULTS = {
  midterm: 40, assignment: 17, attendance: 80, study_hours: 3,
  twelfth_pct: 80, tenth_pct: 80, backlogs: 0, stress: 3,
  distance: 15, complexity: '2', teacher_feedback: '2', participation: '2',
  prev_prev_gpa: 0, intro_grade: 5, hw_grade: 5,
}

export default function PredictionForm({ onResult, onError, isLoading, setLoading }) {
  const [values, setValues] = useState(DEFAULTS)

  const handleChange = (key, val) => setValues(v => ({ ...v, [key]: val }))

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
