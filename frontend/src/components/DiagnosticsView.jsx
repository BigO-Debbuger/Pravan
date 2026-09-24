import React from 'react';
import { Layers, CheckCircle2, AlertCircle, BarChart2, Cpu, Database } from 'lucide-react';

export default function DiagnosticsView({ metadata }) {
  const valMetrics = metadata?.model_metrics?.validation || {};
  const testMetrics = metadata?.model_metrics?.test || {};

  const featureImportances = [
    { name: 'month', gain: 0.428, desc: 'Calendar month (seasonal cycle)' },
    { name: 'month_sin', gain: 0.244, desc: 'Cyclic harmonic sine encoding' },
    { name: 'anomaly_lag3', gain: 0.136, desc: 'Precipitation anomaly 3 months prior' },
    { name: 'lat_norm', gain: 0.053, desc: 'Spatial latitude coordinate normalized' },
    { name: 'anomaly_lag2', gain: 0.047, desc: 'Precipitation anomaly 2 months prior' },
    { name: 'climatology_mm', gain: 0.025, desc: 'Monthly baseline climatology' },
    { name: 'rolling_mean_6m', gain: 0.020, desc: '6-month rolling rainfall mean' },
    { name: 'anomaly_lag1', gain: 0.016, desc: 'Precipitation anomaly 1 month prior' },
    { name: 'rolling_std_3m', gain: 0.014, desc: '3-month rolling rainfall standard dev' },
    { name: 'lon_norm', gain: 0.010, desc: 'Spatial longitude coordinate normalized' }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Title */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Layers size={22} color="var(--accent-cyan)" />
          <div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
              XGBoost Model Architecture & Evaluation Diagnostics
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px', margin: 0 }}>
              Verified spatiotemporal evaluation on monsoon-aware chronological splits
            </p>
          </div>
        </div>
      </div>

      {/* Metrics Comparison Table */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#f8fafc', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BarChart2 size={18} color="var(--accent-teal)" /> Out-of-Time Generalization Metrics
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)', textAlign: 'left' }}>
                <th style={{ padding: '10px 14px' }}>Metric</th>
                <th style={{ padding: '10px 14px' }}>Validation Set (Jun - Jul 2022)</th>
                <th style={{ padding: '10px 14px' }}>Test Set (Aug - Sep 2022, Held-Out)</th>
                <th style={{ padding: '10px 14px' }}>Interpretation</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <td style={{ padding: '12px 14px', fontWeight: '600', color: '#f8fafc' }}>Samples (rows)</td>
                <td style={{ padding: '12px 14px', color: '#38bdf8' }}>30,250 (2 months)</td>
                <td style={{ padding: '12px 14px', color: '#38bdf8' }}>30,250 (2 months)</td>
                <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>15,125 grid cells per month over India</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <td style={{ padding: '12px 14px', fontWeight: '600', color: '#f8fafc' }}>MAE (Mean Absolute Error)</td>
                <td style={{ padding: '12px 14px', color: '#f8fafc' }}>59.06 mm</td>
                <td style={{ padding: '12px 14px', color: '#f8fafc' }}>48.48 mm</td>
                <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>Lower on late monsoon held-out split</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <td style={{ padding: '12px 14px', fontWeight: '600', color: '#f8fafc' }}>RMSE (Root Mean Squared Error)</td>
                <td style={{ padding: '12px 14px', color: '#f8fafc' }}>89.57 mm</td>
                <td style={{ padding: '12px 14px', color: '#f8fafc' }}>70.69 mm</td>
                <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>Standard deviation of residuals</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <td style={{ padding: '12px 14px', fontWeight: '600', color: '#f8fafc' }}>Coefficient of Determination (R²)</td>
                <td style={{ padding: '12px 14px', color: '#f43f5e' }}>-0.0043</td>
                <td style={{ padding: '12px 14px', color: '#f43f5e' }}>-0.0169</td>
                <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>Mean-collapse behavior on 1-year training record</td>
              </tr>
              <tr>
                <td style={{ padding: '12px 14px', fontWeight: '600', color: '#f8fafc' }}>Prediction Std / Actual Std</td>
                <td style={{ padding: '12px 14px', color: 'var(--accent-amber)' }}>0.0138</td>
                <td style={{ padding: '12px 14px', color: 'var(--accent-amber)' }}>0.0448</td>
                <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>Conservative regression toward regional mean</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Importance & Model Params Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
        {/* Feature Importance Bar Chart */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#f8fafc', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={18} color="var(--accent-cyan)" /> Top Predictive Features (Gain)
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {featureImportances.map((f, i) => (
              <div key={i}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '3px' }}>
                  <span style={{ color: '#f8fafc', fontWeight: '500' }}>{f.name}</span>
                  <span style={{ color: 'var(--accent-cyan)', fontWeight: '600' }}>{(f.gain * 100).toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', height: '7px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${f.gain * 100 * 2.2}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #0284c7 0%, #38bdf8 100%)',
                    borderRadius: '4px'
                  }} />
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '2px' }}>{f.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Training Setup & Technical Decisions */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#f8fafc', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Database size={18} color="var(--accent-indigo)" /> Training Configuration
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '0.82rem' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <strong style={{ color: '#38bdf8' }}>Chronological Split (Zero Leakage)</strong>
              <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0' }}>
                Strict time-based ordering: Train (Dec 2020 - May 2022) → Val (Jun - Jul 2022) → Test (Aug - Sep 2022). No random shuffling.
              </p>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <strong style={{ color: '#38bdf8' }}>StandardScaler Discipline</strong>
              <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0' }}>
                Mean and variance statistics computed strictly on the training partition and saved to scaler parameters to prevent validation distribution leakage.
              </p>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <strong style={{ color: '#38bdf8' }}>Early Stopping Guard</strong>
              <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0' }}>
                Early stopping with 30 patience rounds halted training at step 0 to prevent severe overfitting onto the single 2021 monsoon season in training.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
