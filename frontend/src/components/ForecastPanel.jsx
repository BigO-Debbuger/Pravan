import React, { useState } from 'react';
import { TrendingUp, AlertTriangle, ShieldCheck, Sparkles, HelpCircle } from 'lucide-react';

export default function ForecastPanel({ forecastData, loading, onFetchForecast }) {
  const [months, setMonths] = useState(3);

  const getRiskBadge = (category) => {
    switch (category) {
      case 'Excess Rainfall':
        return <span className="badge badge-cyan">Excess (+25mm)</span>;
      case 'Normal':
        return <span className="badge badge-emerald">Normal (±25mm)</span>;
      case 'Moderate Deficit':
        return <span className="badge badge-amber">Moderate Deficit</span>;
      case 'Severe Deficit':
        return <span className="badge badge-rose">Severe Deficit</span>;
      default:
        return <span className="badge badge-cyan">{category}</span>;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header controls */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sparkles size={20} color="var(--accent-cyan)" />
            <h2 style={{ fontSize: '1.15rem', color: '#f8fafc', fontWeight: '700', margin: 0 }}>
              AI Precipitation Anomaly Forecast (Post-Sep 2022)
            </h2>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px', margin: 0 }}>
            Recursive multi-step forecasting via trained XGBoost Baseline Regressor
          </p>
        </div>

        {/* Steps selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Forecast Horizon:</span>
          <div style={{ display: 'flex', gap: '6px' }}>
            {[1, 2, 3, 4, 5, 6].map((num) => (
              <button
                key={num}
                className={`btn btn-outline ${months === num ? 'btn-active' : ''}`}
                onClick={() => {
                  setMonths(num);
                  onFetchForecast(num);
                }}
                style={{ fontSize: '0.8rem', padding: '4px 10px' }}
              >
                {num}M
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading ? (
        <div className="glass-panel" style={{ padding: '40px', textAlign: 'center' }}>
          <p style={{ color: 'var(--text-secondary)' }}>Generating AI forecast vectors...</p>
        </div>
      ) : forecastData && forecastData.forecasts ? (
        <>
          {/* Forecast Cards Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px' }}>
            {forecastData.forecasts.map((fc, idx) => (
              <div
                key={idx}
                className="glass-panel"
                style={{
                  padding: '18px',
                  borderTop: `4px solid ${fc.forecast_anomaly_mm >= 0 ? '#38bdf8' : '#f43f5e'}`,
                  position: 'relative'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <div>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Step +{idx + 1}
                    </span>
                    <h3 style={{ fontSize: '1.1rem', fontWeight: '700', color: '#f8fafc', margin: '2px 0 0 0' }}>
                      {fc.month_name} {fc.date.slice(0, 4)}
                    </h3>
                  </div>
                  {getRiskBadge(fc.risk_category)}
                </div>

                {/* Metrics */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: 'rgba(15, 23, 42, 0.4)', padding: '12px', borderRadius: '8px', marginBottom: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Predicted Anomaly:</span>
                    <strong style={{ color: fc.forecast_anomaly_mm >= 0 ? '#38bdf8' : '#f43f5e' }}>
                      {fc.forecast_anomaly_mm > 0 ? `+${fc.forecast_anomaly_mm}` : fc.forecast_anomaly_mm} mm
                    </strong>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Baseline Climatology:</span>
                    <strong style={{ color: '#f8fafc' }}>
                      {fc.climatology_mm} mm
                    </strong>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '6px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Total Forecast Rain:</span>
                    <strong style={{ color: '#38bdf8', fontSize: '0.95rem' }}>
                      {fc.forecast_total_mm} mm
                    </strong>
                  </div>
                </div>

                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  Target date: {fc.date}
                </div>
              </div>
            ))}
          </div>

          {/* Scientific Disclaimer Card */}
          <div className="glass-panel" style={{
            padding: '18px 22px',
            borderLeft: '4px solid var(--accent-amber)',
            background: 'rgba(245, 158, 11, 0.05)'
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
              <AlertTriangle size={20} color="var(--accent-amber)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <h4 style={{ fontSize: '0.92rem', fontWeight: '600', color: '#f8fafc', marginBottom: '4px' }}>
                  Scientific & Academic Limitation Notice
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                  The training dataset encompasses 28 monthly timesteps from ERA5 reanalysis (with only 18 usable training months containing a single prior monsoon season). As scientifically diagnosed during baseline evaluation, the XGBoost model regresses predictions toward the regional mean (pred_std / actual_std ≈ 0.014–0.045) rather than overconfidently predicting extreme variance. This model serves as an initial spatiotemporal ML demonstration and should be interpreted alongside physical meteorological guidance.
                </p>
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
