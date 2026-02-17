import React, { createContext, useContext, useState, useEffect } from 'react';
import { API_URL } from '../config';

const StatusContext = createContext();

export const StatusProvider = ({ children }) => {
    const [status, setStatus] = useState({
        pipeline: 'idle',
        mlflow: 'idle',
        prefect: 'idle'
    });

    const fetchStatus = async () => {
        try {
            const res = await fetch(`${API_URL}/status`);
            const data = await res.json();
            setStatus({
                pipeline: data.pipeline || 'idle',
                mlflow: data.mlflow || 'idle',
                prefect: data.prefect || 'idle'
            });
        } catch (e) {
            console.error("Status check failed:", e);
        }
    };

    useEffect(() => {
        fetchStatus(); // Initial fetch
        const interval = setInterval(fetchStatus, 3000);
        return () => clearInterval(interval);
    }, []);

    return (
        <StatusContext.Provider value={{ status, refreshStatus: fetchStatus }}>
            {children}
        </StatusContext.Provider>
    );
};

export const useStatus = () => {
    const context = useContext(StatusContext);
    if (!context) {
        throw new Error('useStatus must be used within a StatusProvider');
    }
    return context;
};
