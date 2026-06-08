export default function HeroSection({ role }) {
  return (
    <div className="hero">
      <div className="hero-tag">
        🏫 Multi-Modal ML · Audio + Vision + Survey · Stacking Ensemble
      </div>
      <h1>
        {role === 'teacher'
          ? <>Batch CGPA Prediction<br />Teacher Dashboard</>
          : <>Predict Your Semester CGPA<br />with Multi-Modal Machine Learning</>
        }
      </h1>
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
  )
}
