const API_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const MLFLOW_URL = import.meta.env.VITE_MLFLOW_URL || 'http://localhost:5000';
const PREFECT_URL = import.meta.env.VITE_PREFECT_URL || 'http://localhost:4200';

export { API_URL, MLFLOW_URL, PREFECT_URL };
