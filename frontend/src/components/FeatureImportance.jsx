import { useEffect, useState } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const COLORS = [
  '#6366f1','#818cf8','#06b6d4','#22c55e','#f59e0b','#c084fc','#f87171',
  '#34d399','#60a5fa','#fb923c','#e879f9','#a3e635','#38bdf8','#fbbf24',
  '#4ade80','#f472b6','#a78bfa','#fb7185',
]

export default function FeatureImportance() {
  const [data, setData]     = useState([])
  const [loading, setLoad]  = useState(true)
  const [error, setError]   = useState(null)

  useEffect(() => {
    axios.get('/api/feature-importance')
      .then(r => {
        const top10 = r.data.features.slice(0, 10).map(f => ({
          name: f.display_name,
          value: +(f.importance * 100).toFixed(2),
        }))
        setData(top10)
      })
      .catch(() => setError('Could not load feature importance'))
      .finally(() => setLoad(false))
  }, [])

  if (loading) return (
    <div className="card">
      <div className="card-title"><span>📊</span> Feature Importance</div>
      <div className="empty-state"><div className="spinner" style={{ width:24, height:24 }} /></div>
    </div>
  )

  if (error) return (
    <div className="card">
      <div className="card-title"><span>📊</span> Feature Importance</div>
      <div className="empty-state"><p>{error}</p></div>
    </div>
  )

  return (
    <div className="card">
      <div className="card-title"><span>📊</span> Feature Importance — Top 10 Predictors</div>
      <div className="fi-chart-wrapper">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
            <XAxis
              type="number"
              tick={{ fill: '#64748b', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={v => `${v}%`}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={130}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              formatter={(v) => [`${v}%`, 'Importance']}
              contentStyle={{ background: '#141d35', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: '#f1f5f9' }}
              itemStyle={{ color: '#818cf8' }}
            />
            <Bar dataKey="value" radius={[0, 6, 6, 0]} maxBarSize={18}>
              {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p style={{ fontSize:'0.73rem', color:'var(--text3)', marginTop:'0.75rem', lineHeight:1.5 }}>
        Importance = contribution to reducing prediction error across all trees in the ensemble.
        Higher % → stronger predictor of CGPA.
      </p>
    </div>
  )
}
