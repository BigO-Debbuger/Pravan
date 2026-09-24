import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, CircleMarker, Marker, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import { Calendar, Eye, MapPin } from 'lucide-react';

// Custom Pin icon
const createPinIcon = () => {
  return L.divIcon({
    className: 'custom-pin',
    html: `<div style="
      width: 18px; 
      height: 18px; 
      border-radius: 50%; 
      background: #38bdf8; 
      border: 3px solid #ffffff; 
      box-shadow: 0 0 15px #38bdf8;
    "></div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9]
  });
};

function MapClickHandler({ onSelectCoords }) {
  useMapEvents({
    click(e) {
      const lat = Math.round(e.latlng.lat * 100) / 100;
      const lon = Math.round(e.latlng.lng * 100) / 100;
      if (lat >= 6 && lat <= 37 && lon >= 68 && lon <= 98) {
        onSelectCoords(lat, lon);
      }
    }
  });
  return null;
}

export default function MapView({
  gridData,
  selectedDate,
  setSelectedDate,
  availableDates,
  selectedLat,
  selectedLon,
  onSelectCoords,
  loadingGrid
}) {
  const getColor = (val) => {
    if (val >= 100) return '#0284c7';
    if (val >= 40) return '#38bdf8';
    if (val >= 10) return '#7dd3fc';
    if (val > -10) return '#94a3b8';
    if (val > -40) return '#fcd34d';
    if (val > -100) return '#f97316';
    return '#ef4444';
  };

  return (
    <div className="glass-panel" style={{ padding: '18px', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Map Header with Date Switcher */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '14px',
        flexWrap: 'wrap',
        gap: '10px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <MapPin size={18} color="var(--accent-cyan)" />
          <h2 style={{ fontSize: '1.05rem', color: '#f8fafc', fontWeight: '600' }}>
            Spatial Precipitation Anomaly Grid
          </h2>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(255,255,255,0.05)', padding: '4px 10px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <Calendar size={14} color="var(--text-secondary)" />
            <select
              value={selectedDate || (availableDates.length > 0 ? availableDates[availableDates.length - 1] : '')}
              onChange={(e) => setSelectedDate(e.target.value)}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#f8fafc',
                fontSize: '0.8rem',
                fontFamily: 'inherit',
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              {availableDates.map(d => (
                <option key={d} value={d} style={{ background: '#111a2e', color: '#f8fafc' }}>
                  {d}
                </option>
              ))}
            </select>
          </div>
          {loadingGrid && (
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>Loading grid...</span>
          )}
        </div>
      </div>

      {/* Map Container */}
      <div style={{ flex: 1, minHeight: '460px', position: 'relative', borderRadius: '10px', overflow: 'hidden' }}>
        <MapContainer
          center={[22.5, 82.5]}
          zoom={5}
          minZoom={4}
          maxZoom={9}
          style={{ width: '100%', height: '100%' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />

          <MapClickHandler onSelectCoords={onSelectCoords} />

          {/* Anomaly Grid Points */}
          {gridData && gridData.points && gridData.points.map((pt, idx) => (
            <CircleMarker
              key={idx}
              center={[pt.lat, pt.lon]}
              radius={3}
              pathOptions={{
                fillColor: getColor(pt.val),
                color: getColor(pt.val),
                weight: 0,
                fillOpacity: 0.65
              }}
            >
              <Popup>
                <div style={{ fontSize: '0.75rem', lineHeight: '1.4' }}>
                  <strong>Grid Cell:</strong> {pt.lat}°N, {pt.lon}°E<br />
                  <strong>Anomaly:</strong> {pt.val > 0 ? `+${pt.val}` : pt.val} mm<br />
                  <span style={{ color: getColor(pt.val), fontWeight: '600' }}>
                    {pt.val >= 40 ? 'Wet Anomaly' : pt.val <= -40 ? 'Deficit Anomaly' : 'Near Normal'}
                  </span>
                </div>
              </Popup>
            </CircleMarker>
          ))}

          {/* Selected Location Marker */}
          {selectedLat && selectedLon && (
            <Marker position={[selectedLat, selectedLon]} icon={createPinIcon()}>
              <Popup>
                <div style={{ fontSize: '0.8rem' }}>
                  <strong>Target Location</strong><br />
                  Lat: {selectedLat}°N, Lon: {selectedLon}°E
                </div>
              </Popup>
            </Marker>
          )}
        </MapContainer>

        {/* Legend Overlay */}
        <div style={{
          position: 'absolute',
          bottom: '16px',
          left: '16px',
          zIndex: 1000,
          background: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(8px)',
          padding: '10px 14px',
          borderRadius: '8px',
          border: '1px solid var(--border-color)',
          fontSize: '0.72rem',
          color: 'var(--text-secondary)'
        }}>
          <div style={{ fontWeight: '600', color: '#f8fafc', marginBottom: '6px' }}>Anomaly Scale (mm)</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>-100+</span>
            <div style={{
              width: '120px',
              height: '8px',
              borderRadius: '4px',
              background: 'linear-gradient(to right, #ef4444, #f97316, #fcd34d, #94a3b8, #7dd3fc, #38bdf8, #0284c7)'
            }}></div>
            <span>+100+</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '0.65rem' }}>
            <span style={{ color: '#ef4444' }}>Deficit</span>
            <span style={{ color: '#94a3b8' }}>Normal</span>
            <span style={{ color: '#38bdf8' }}>Excess</span>
          </div>
        </div>

        {/* Hint banner */}
        <div style={{
          position: 'absolute',
          top: '12px',
          right: '12px',
          zIndex: 1000,
          background: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(8px)',
          padding: '6px 12px',
          borderRadius: '6px',
          border: '1px solid var(--border-color)',
          fontSize: '0.72rem',
          color: 'var(--text-secondary)'
        }}>
          💡 Click anywhere on India to inspect time series
        </div>
      </div>
    </div>
  );
}
