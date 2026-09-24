import React, { useState } from 'react';
import { BarChart3, TrendingUp, Calendar, AlertCircle } from 'lucide-react';

export default function ChartsView({ anomalyData, climatologyData, loading }) {
  const [hoveredIdx, setHoveredIdx] = useState(null);

  if (loading) {
    return (
      <div className="glass-panel" style={{ padding: '30px', textAlign: 'center' }}>
        <p style={{ color: 'var(--text-secondary)' }}>Loading time-series analytics...</p>
      </div>
    );
  }

  if (!anomalyData || !anomalyData.history || anomalyData.history.length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '30px', textAlign: 'center' }}>
        <p style={{ color: 'var(--text-secondary)' }}>No time-series data available for this cell.</p>
      </div>
    );
  }

  const history = anomalyData.history;
  const maxRain = Math.max(...history.map(h => Math.max(h.precipitation_mm, h.climatology_mm, 10)));
  const maxAnom = Math.max(...history.map(h => Math.abs(h.anomaly_mm)), 20);

  // SVG Chart dimensions
  const width = 680;
  const height = 180;
  const padding = { top: 20, right: 20, bottom: 30, left: 45 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  // Scale helpers
  const getX = (idx) => padding.left + (idx / (history.length - 1)) * chartW;
  const getYRain = (val) => padding.top + chartH - (val / maxRain) * chartH;
  const getYAnom = (val) => padding.top + (chartH / 2) - (val / (maxAnom * 1.1)) * (chartH / 2);

  // Line paths
  const rainPath = history.reduce((acc, h, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${getX(i)} ${getYRain(h.precipitation_mm)}`, '');
  const climPath = history.reduce((acc, h, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${getX(i)} ${getYRain(h.climatology_mm)}`, '');
  const areaRainPath = `${rainPath} L ${getX(history.length - 1)} ${padding.top + chartH} L ${padding.left} ${padding.top + chartH} Z`;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Chart 1: Rainfall vs Baseline Climatology */}
      <div className="glass-panel" style={{ padding: '18px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingUp size={18} color="var(--accent-cyan)" />
            <h3 style={{ fontSize: '0.95rem', fontWeight: '600', color: '#f8fafc', margin: 0 }}>
              Monthly Rainfall vs. Climatology Baseline (2020–2022)
            </h3>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px', fontSize: '0.75rem' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: '#38bdf8' }}>
              <span style={{ width: '10px', height: '3px', background: '#38bdf8', borderRadius: '2px' }} /> Actual Rainfall
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: '#94a3b8' }}>
              <span style={{ width: '10px', height: '2px', borderTop: '2px dashed #94a3b8' }} /> Baseline Climatology
            </span>
          </div>
        </div>

        {/* Hover info badge */}
        <div style={{ minHeight: '22px', marginBottom: '4px', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
          {hoveredIdx !== null ? (
            <span>
              <strong>{history[hoveredIdx].date}</strong> | Rainfall: <span style={{ color: '#38bdf8', fontWeight: '600' }}>{history[hoveredIdx].precipitation_mm} mm</span> | Baseline: {history[hoveredIdx].climatology_mm} mm | Anomaly: <span style={{ color: history[hoveredIdx].anomaly_mm >= 0 ? '#38bdf8' : '#f43f5e', fontWeight: '600' }}>{history[hoveredIdx].anomaly_mm > 0 ? `+${history[hoveredIdx].anomaly_mm}` : history[hoveredIdx].anomaly_mm} mm</span>
            </span>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>Hover over data points to inspect monthly details</span>
          )}
        </div>

        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', overflow: 'visible' }}>
          {/* Y Axis Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
            const y = padding.top + chartH * (1 - pct);
            const val = Math.round(maxRain * pct);
            return (
              <g key={i}>
                <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                <text x={padding.left - 8} y={y + 4} fill="#64748b" fontSize="9" textAnchor="end">{val}</text>
              </g>
            );
          })}

          {/* Area Fill */}
          <path d={areaRainPath} fill="url(#rainGradient)" opacity="0.3" />
          <defs>
            <linearGradient id="rainGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0284c7" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#0284c7" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Climatology baseline line (dashed) */}
          <path d={climPath} fill="none" stroke="#94a3b8" strokeWidth="1.8" strokeDasharray="4 4" />

          {/* Actual Rainfall line */}
          <path d={rainPath} fill="none" stroke="#38bdf8" strokeWidth="2.2" />

          {/* Data Points */}
          {history.map((h, i) => (
            <circle
              key={i}
              cx={getX(i)}
              cy={getYRain(h.precipitation_mm)}
              r={hoveredIdx === i ? 5 : 2.5}
              fill={hoveredIdx === i ? '#ffffff' : '#38bdf8'}
              stroke="#0284c7"
              strokeWidth={hoveredIdx === i ? 2 : 1}
              style={{ cursor: 'pointer', transition: 'all 0.15s ease' }}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
            />
          ))}

          {/* X Axis Labels */}
          {history.filter((_, i) => i % 4 === 0 || i === history.length - 1).map((h, idx) => (
            <text
              key={idx}
              x={getX(history.indexOf(h))}
              y={height - 6}
              fill="#64748b"
              fontSize="9"
              textAnchor="middle"
            >
              {h.date.slice(2, 7)}
            </text>
          ))}
        </svg>
      </div>

      {/* Chart 2: Precipitation Anomaly Bars */}
      <div className="glass-panel" style={{ padding: '18px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={18} color="var(--accent-teal)" />
            <h3 style={{ fontSize: '0.95rem', fontWeight: '600', color: '#f8fafc', margin: 0 }}>
              Precipitation Anomaly Deviations (mm)
            </h3>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.75rem' }}>
            <span style={{ color: '#38bdf8' }}>■ Wet (+ Anomaly)</span>
            <span style={{ color: '#f43f5e' }}>■ Deficit (- Anomaly)</span>
          </div>
        </div>

        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto' }}>
          {/* Zero baseline */}
          <line
            x1={padding.left}
            y1={getYAnom(0)}
            x2={width - padding.right}
            y2={getYAnom(0)}
            stroke="rgba(255,255,255,0.25)"
            strokeWidth="1.2"
          />

          {/* Anomaly Bars */}
          {history.map((h, i) => {
            const x = getX(i) - 6;
            const zeroY = getYAnom(0);
            const valY = getYAnom(h.anomaly_mm);
            const barH = Math.max(2, Math.abs(valY - zeroY));
            const y = h.anomaly_mm >= 0 ? valY : zeroY;
            const color = h.anomaly_mm >= 0 ? '#38bdf8' : '#f43f5e';

            return (
              <g key={i}>
                <rect
                  x={x}
                  y={y}
                  width="12"
                  height={barH}
                  fill={color}
                  rx="2"
                  opacity={hoveredIdx === i ? 1 : 0.8}
                  style={{ cursor: 'pointer', transition: 'all 0.15s ease' }}
                  onMouseEnter={() => setHoveredIdx(i)}
                  onMouseLeave={() => setHoveredIdx(null)}
                />
              </g>
            );
          })}

          {/* X Axis Labels */}
          {history.filter((_, i) => i % 4 === 0 || i === history.length - 1).map((h, idx) => (
            <text
              key={idx}
              x={getX(history.indexOf(h))}
              y={height - 6}
              fill="#64748b"
              fontSize="9"
              textAnchor="middle"
            >
              {h.date.slice(2, 7)}
            </text>
          ))}
        </svg>
      </div>

      {/* Chart 3: Annual Climatology Cycle */}
      {climatologyData && climatologyData.monthly_climatology && (
        <div className="glass-panel" style={{ padding: '16px 18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <h4 style={{ fontSize: '0.85rem', fontWeight: '600', color: '#f8fafc', margin: 0 }}>
              12-Month Baseline Climatology Cycle (Annual Mean: {climatologyData.annual_climatology_total_mm} mm)
            </h4>
            <span className="badge badge-amber" style={{ fontSize: '0.65rem' }}>
              Monsoon Peak: Jun - Sep
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: '6px', marginTop: '10px' }}>
            {climatologyData.monthly_climatology.map((m) => {
              const isMonsoon = [6, 7, 8, 9].includes(m.month);
              return (
                <div
                  key={m.month}
                  style={{
                    background: isMonsoon ? 'rgba(56, 189, 248, 0.12)' : 'rgba(255,255,255,0.03)',
                    border: `1px solid ${isMonsoon ? 'rgba(56, 189, 248, 0.3)' : 'var(--border-color)'}`,
                    borderRadius: '6px',
                    padding: '6px 4px',
                    textAlign: 'center'
                  }}
                >
                  <div style={{ fontSize: '0.7rem', color: isMonsoon ? '#38bdf8' : 'var(--text-secondary)', fontWeight: isMonsoon ? '600' : '400' }}>
                    {m.month_name.slice(0, 3)}
                  </div>
                  <div style={{ fontSize: '0.78rem', color: '#f8fafc', fontWeight: '600', marginTop: '2px' }}>
                    {m.climatology_mm}
                  </div>
                  <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>mm</div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
