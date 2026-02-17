import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Home from './pages/Home';
import Prefect from './pages/Prefect';
import MLflow from './pages/MLflow';
import { StatusProvider } from './context/StatusContext';
import './index.css';

function App() {
  return (
    <StatusProvider>
      <Router>
        <Navigation />
        <main style={{ flexGrow: 1 }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/prefect" element={<Prefect />} />
            <Route path="/mlflow" element={<MLflow />} />
          </Routes>
        </main>
        <footer style={{ padding: '2rem', textAlign: 'center', borderTop: '1px solid var(--glass-border)', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          &copy; 2026 Federated Learning with MLOps Dashboard
        </footer>
      </Router>
    </StatusProvider>
  );
}

export default App;
