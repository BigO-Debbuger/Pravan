import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import MapView from './components/MapView';
import LocationSelector from './components/LocationSelector';
import ChartsView from './components/ChartsView';
import ForecastPanel from './components/ForecastPanel';
import DiagnosticsView from './components/DiagnosticsView';

const API_BASE = '/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // Coordinates (default: New Delhi)
  const [selectedLat, setSelectedLat] = useState(28.61);
  const [selectedLon, setSelectedLon] = useState(77.21);

  // Metadata
  const [metadata, setMetadata] = useState(null);
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState('2022-09-30');

  // Datasets
  const [gridData, setGridData] = useState(null);
  const [anomalyData, setAnomalyData] = useState(null);
  const [climatologyData, setClimatologyData] = useState(null);
  const [forecastData, setForecastData] = useState(null);

  // Loading states
  const [loadingGrid, setLoadingGrid] = useState(false);
  const [loadingSeries, setLoadingSeries] = useState(false);
  const [loadingForecast, setLoadingForecast] = useState(false);

  // 1. Initial Metadata Load
  useEffect(() => {
    fetch(`${API_BASE}/metadata`)
      .then(res => res.json())
      .then(data => {
        setMetadata(data);
        if (data.available_dates && data.available_dates.length > 0) {
          setAvailableDates(data.available_dates);
          setSelectedDate(data.available_dates[data.available_dates.length - 1]);
        }
      })
      .catch(err => console.error('Failed to load metadata:', err));
  }, []);

  // 2. Fetch Spatial Grid when Date changes
  useEffect(() => {
    if (!selectedDate) return;
    setLoadingGrid(true);
    fetch(`${API_BASE}/grid?date=${selectedDate}&step=2`)
      .then(res => res.json())
      .then(data => {
        setGridData(data);
        setLoadingGrid(false);
      })
      .catch(err => {
        console.error('Failed to fetch grid:', err);
        setLoadingGrid(false);
      });
  }, [selectedDate]);

  // 3. Fetch Time Series and Climatology when Coords change
  useEffect(() => {
    setLoadingSeries(true);
    Promise.all([
      fetch(`${API_BASE}/anomaly?lat=${selectedLat}&lon=${selectedLon}`).then(r => r.json()),
      fetch(`${API_BASE}/climatology?lat=${selectedLat}&lon=${selectedLon}`).then(r => r.json())
    ])
      .then(([anom, clim]) => {
        setAnomalyData(anom);
        setClimatologyData(clim);
        setLoadingSeries(false);
      })
      .catch(err => {
        console.error('Failed to fetch series:', err);
        setLoadingSeries(false);
      });

    // Also trigger initial 3-month forecast
    fetchForecast(3);
  }, [selectedLat, selectedLon]);

  // 4. Fetch Forecast helper
  const fetchForecast = (months = 3) => {
    setLoadingForecast(true);
    fetch(`${API_BASE}/forecast?lat=${selectedLat}&lon=${selectedLon}&months=${months}`)
      .then(res => res.json())
      .then(data => {
        setForecastData(data);
        setLoadingForecast(false);
      })
      .catch(err => {
        console.error('Failed to fetch forecast:', err);
        setLoadingForecast(false);
      });
  };

  const handleSelectCoords = (lat, lon) => {
    setSelectedLat(lat);
    setSelectedLon(lon);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        metadata={metadata}
      />

      <main style={{ flex: 1, padding: '0 20px 24px 20px' }}>
        {activeTab === 'dashboard' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <LocationSelector
              presets={metadata?.locations || []}
              selectedLat={selectedLat}
              selectedLon={selectedLon}
              onSelectLocation={handleSelectCoords}
            />

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(350px, 1fr) minmax(400px, 1.3fr)',
              gap: '16px'
            }}>
              <MapView
                gridData={gridData}
                selectedDate={selectedDate}
                setSelectedDate={setSelectedDate}
                availableDates={availableDates}
                selectedLat={selectedLat}
                selectedLon={selectedLon}
                onSelectCoords={handleSelectCoords}
                loadingGrid={loadingGrid}
              />

              <ChartsView
                anomalyData={anomalyData}
                climatologyData={climatologyData}
                loading={loadingSeries}
              />
            </div>
          </div>
        )}

        {activeTab === 'forecast' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <LocationSelector
              presets={metadata?.locations || []}
              selectedLat={selectedLat}
              selectedLon={selectedLon}
              onSelectLocation={handleSelectCoords}
            />

            <ForecastPanel
              forecastData={forecastData}
              loading={loadingForecast}
              onFetchForecast={fetchForecast}
            />
          </div>
        )}

        {activeTab === 'diagnostics' && (
          <DiagnosticsView metadata={metadata} />
        )}
      </main>

      <footer style={{
        textAlign: 'center',
        padding: '16px 20px',
        borderTop: '1px solid var(--border-color)',
        fontSize: '0.75rem',
        color: 'var(--text-muted)'
      }}>
        Smart India Hackathon (SIH) — Precipitation Anomaly & Forecasting System | Powered by ECMWF ERA5 Reanalysis & XGBoost ML
      </footer>
    </div>
  );
}
