import React from 'react';
import { CloudRain, Activity, Layers, TrendingUp, Info } from 'lucide-react';

export default function Header({ activeTab, setActiveTab, metadata, systemStatus }) {
  return (
    <header className="glass-panel" style={{
      margin: '16px 20px',
      padding: '14px 24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '16px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{
          width: '42px',
          height: '42px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #0284c7 0%, #38bdf8 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 20px rgba(56, 189, 248, 0.4)'
        }}>
          <CloudRain size={24} color="#ffffff" />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
              VarshaVani
            </h1>
            <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>SIH-2024</span>
            <span className="badge badge-emerald" style={{ fontSize: '0.65rem' }}>ERA5 Reanalysis</span>
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '2px 0 0 0' }}>
            India Precipitation Climatology, Anomaly Detection & ML Forecasting
          </p>
        </div>
      </div>

      {/* Tabs */}
      <nav style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(15, 23, 42, 0.5)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
        <button
          className={`btn btn-outline ${activeTab === 'dashboard' ? 'btn-active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
          style={{ fontSize: '0.82rem', padding: '6px 14px' }}
        >
          <Activity size={15} /> Dashboard & Map
        </button>
        <button
          className={`btn btn-outline ${activeTab === 'forecast' ? 'btn-active' : ''}`}
          onClick={() => setActiveTab('forecast')}
          style={{ fontSize: '0.82rem', padding: '6px 14px' }}
        >
          <TrendingUp size={15} /> AI Forecast
        </button>
        <button
          className={`btn btn-outline ${activeTab === 'diagnostics' ? 'btn-active' : ''}`}
          onClick={() => setActiveTab('diagnostics')}
          style={{ fontSize: '0.82rem', padding: '6px 14px' }}
        >
          <Layers size={15} /> Model Diagnostics
        </button>
      </nav>

      {/* Live Status indicator */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 12px',
          background: 'rgba(16, 185, 129, 0.08)',
          border: '1px solid rgba(16, 185, 129, 0.25)',
          borderRadius: '8px',
          fontSize: '0.75rem',
          color: '#34d399'
        }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px #10b981' }} />
          <span>API Connected (0.25° India)</span>
        </div>
      </div>
    </header>
  );
}
