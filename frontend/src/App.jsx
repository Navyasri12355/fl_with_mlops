import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navigation from './components/Navigation';
import Home from './pages/Home';
import Prefect from './pages/Prefect';
import MLflow from './pages/MLflow';
import Login from './pages/Login';
import Signup from './pages/Signup';
import { StatusProvider } from './context/StatusContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import './index.css';

const ProtectedRoute = ({ children }) => {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" />;
  return children;
};

const PublicRoute = ({ children }) => {
  const { user } = useAuth();
  if (user) return <Navigate to="/" />;
  return children;
};

function App() {
  return (
    <AuthProvider>
      <StatusProvider>
        <Router>
          <Navigation />
          <main style={{ flexGrow: 1 }}>
            <Routes>
              <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
              <Route path="/signup" element={<PublicRoute><Signup /></PublicRoute>} />
              <Route path="/" element={<ProtectedRoute><Home /></ProtectedRoute>} />
              <Route path="/prefect" element={<ProtectedRoute><Prefect /></ProtectedRoute>} />
              <Route path="/mlflow" element={<ProtectedRoute><MLflow /></ProtectedRoute>} />
            </Routes>
          </main>
          <footer style={{ padding: '2rem', textAlign: 'center', borderTop: '1px solid var(--glass-border)', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            &copy; 2026 Federated Learning with MLOps Dashboard
          </footer>
        </Router>
      </StatusProvider>
    </AuthProvider>
  );
}

export default App;
