import { useState, useRef } from 'react'
import axios from 'axios'

export default function BatchUpload() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef()

  const handleFile = (f) => {
    if (f && f.name.toLowerCase().endsWith('.csv')) {
      setFile(f)
      setError(null)
      setResult(null)
    } else {
      setError('Please select a .csv file')
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0])
  }

  const handleSubmit = async () => {
    if (!file) return
    setUploading(true)
    setError(null)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await axios.post('/api/batch-predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
      })

      const totalRows   = parseInt(res.headers['x-total-rows'] || '0')
      const originalCGPA = parseInt(res.headers['x-original-cgpa'] || '0')
      const predictedCGPA= parseInt(res.headers['x-predicted-cgpa'] || '0')

      // Create download link
      const blob = new Blob([res.data], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)

      setResult({ totalRows, originalCGPA, predictedCGPA, downloadUrl: url })
    } catch (err) {
      if (err.response?.data) {
        // Error might be blob, try to read it
        try {
          const text = await err.response.data.text()
          const parsed = JSON.parse(text)
          setError(parsed.detail || 'Upload failed')
        } catch {
          setError('Upload failed. Make sure the CSV format matches original_data.csv')
        }
      } else {
        setError('Upload failed. Is the API server running?')
      }
    } finally {
      setUploading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="card batch-card">
      <div className="card-title"><span>👩‍🏫</span> Teacher — Batch CGPA Prediction</div>
      <p className="batch-desc">
        Upload a raw student survey CSV file. The system will automatically clean, process, and predict missing CGPA values using the trained Stacking Ensemble model.
      </p>

      {/* Upload Area */}
      {!result && (
        <>
          <div
            className={`drop-zone ${dragOver ? 'drop-zone-active' : ''} ${file ? 'drop-zone-has-file' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
            {file ? (
              <>
                <div className="drop-icon">📄</div>
                <div className="drop-filename">{file.name}</div>
                <div className="drop-filesize">{(file.size / 1024).toFixed(1)} KB</div>
                <div className="drop-hint">Click to change file</div>
              </>
            ) : (
              <>
                <div className="drop-icon">📂</div>
                <div className="drop-text">Drag & Drop your CSV here</div>
                <div className="drop-hint">or click to browse • .csv format only</div>
              </>
            )}
          </div>

          <div className="batch-info">
            <div className="batch-info-item">
              <span className="batch-info-icon">🔧</span>
              <span>Auto-cleans messy text data</span>
            </div>
            <div className="batch-info-item">
              <span className="batch-info-icon">🧠</span>
              <span>Predicts missing CGPA using ML</span>
            </div>
            <div className="batch-info-item">
              <span className="batch-info-icon">📊</span>
              <span>Returns complete dataset</span>
            </div>
          </div>

          {error && <div className="batch-error">{error}</div>}

          <button
            className="submit-btn"
            onClick={handleSubmit}
            disabled={!file || uploading}
            style={{ marginTop: '1rem' }}
          >
            {uploading ? (
              <><div className="spinner" /> Processing…</>
            ) : (
              <> 🚀 Process & Predict</>
            )}
          </button>
        </>
      )}

      {/* Result */}
      {result && (
        <div className="batch-result">
          <div className="batch-result-header">✅ Batch Prediction Complete!</div>

          <div className="batch-stats">
            <div className="batch-stat">
              <div className="batch-stat-val">{result.totalRows}</div>
              <div className="batch-stat-label">Total Students</div>
            </div>
            <div className="batch-stat batch-stat-green">
              <div className="batch-stat-val">{result.originalCGPA}</div>
              <div className="batch-stat-label">Had CGPA</div>
            </div>
            <div className="batch-stat batch-stat-purple">
              <div className="batch-stat-val">{result.predictedCGPA}</div>
              <div className="batch-stat-label">CGPA Predicted</div>
            </div>
            <div className="batch-stat batch-stat-blue">
              <div className="batch-stat-val">
                {result.totalRows > 0 ? ((result.predictedCGPA / result.totalRows) * 100).toFixed(1) : 0}%
              </div>
              <div className="batch-stat-label">Fill Rate</div>
            </div>
          </div>

          <p className="batch-result-note">
            The output CSV has two new columns: <strong>Predicted_CGPA</strong> (the final value) and <strong>Was_Predicted</strong> (Yes/No — whether the model filled it).
          </p>

          <div className="batch-actions">
            <a href={result.downloadUrl} download={`predicted_${file?.name || 'data.csv'}`} className="download-btn">
              ⬇️ Download Completed CSV
            </a>
            <button className="reset-btn" onClick={handleReset}>
              🔄 Upload Another
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
