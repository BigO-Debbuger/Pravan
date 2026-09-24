import React, { useState } from 'react';
import { MapPin, Navigation, Compass } from 'lucide-react';

export default function LocationSelector({
  presets,
  selectedLat,
  selectedLon,
  onSelectLocation
}) {
  const [customLat, setCustomLat] = useState(selectedLat.toString());
  const [customLon, setCustomLon] = useState(selectedLon.toString());

  const handleApply = (e) => {
    e.preventDefault();
    const lat = parseFloat(customLat);
    const lon = parseFloat(customLon);
    if (!isNaN(lat) && !isNaN(lon)) {
      onSelectLocation(lat, lon);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '16px', marginBottom: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Compass size={18} color="var(--accent-cyan)" />
          <h3 style={{ fontSize: '0.95rem', fontWeight: '600', color: '#f8fafc', margin: 0 }}>
            Location Query Coordinates
          </h3>
        </div>

        {/* Current Active Coordinates readout */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Selected:</span>
          <span className="badge badge-cyan" style={{ fontSize: '0.75rem', padding: '3px 8px' }}>
            {selectedLat.toFixed(2)}°N, {selectedLon.toFixed(2)}°E
          </span>
        </div>
      </div>

      {/* Preset Cities Pills */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        overflowX: 'auto',
        paddingBottom: '8px',
        marginBottom: '12px'
      }}>
        {presets && presets.map((loc) => {
          const isSelected = Math.abs(loc.lat - selectedLat) < 0.2 && Math.abs(loc.lon - selectedLon) < 0.2;
          return (
            <button
              key={loc.name}
              className={`btn btn-outline ${isSelected ? 'btn-active' : ''}`}
              onClick={() => {
                setCustomLat(loc.lat.toString());
                setCustomLon(loc.lon.toString());
                onSelectLocation(loc.lat, loc.lon);
              }}
              style={{
                fontSize: '0.75rem',
                padding: '4px 10px',
                whiteSpace: 'nowrap',
                borderRadius: '6px'
              }}
            >
              <MapPin size={12} /> {loc.name}
            </button>
          );
        })}
      </div>

      {/* Custom Coordinate Form */}
      <form onSubmit={handleApply} style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        flexWrap: 'wrap',
        background: 'rgba(15, 23, 42, 0.4)',
        padding: '8px 12px',
        borderRadius: '8px',
        border: '1px solid var(--border-color)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Lat (°N):</label>
          <input
            type="number"
            step="0.05"
            min="6.0"
            max="37.0"
            value={customLat}
            onChange={(e) => setCustomLat(e.target.value)}
            style={{
              width: '80px',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid var(--border-color)',
              background: '#111a2e',
              color: '#f8fafc',
              fontSize: '0.8rem',
              outline: 'none'
            }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Lon (°E):</label>
          <input
            type="number"
            step="0.05"
            min="68.0"
            max="98.0"
            value={customLon}
            onChange={(e) => setCustomLon(e.target.value)}
            style={{
              width: '80px',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid var(--border-color)',
              background: '#111a2e',
              color: '#f8fafc',
              fontSize: '0.8rem',
              outline: 'none'
            }}
          />
        </div>

        <button
          type="submit"
          className="btn btn-primary"
          style={{ fontSize: '0.78rem', padding: '4px 12px' }}
        >
          <Navigation size={12} /> Query Cell
        </button>

        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
          Valid bounds: 6°N-37°N, 68°E-98°E
        </span>
      </form>
    </div>
  );
}
