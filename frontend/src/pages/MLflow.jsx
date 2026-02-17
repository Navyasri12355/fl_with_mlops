import React from 'react';
import { motion } from 'framer-motion';
import { LineChart, Play, Square, ExternalLink, BarChart2, Tag, Layers } from 'lucide-react';
import { useStatus } from '../context/StatusContext';
import { API_URL, MLFLOW_URL } from '../config';

const MLflow = () => {
    const { status: globalStatus, refreshStatus } = useStatus();
    const status = globalStatus.mlflow;

    const startServer = async () => {
        try {
            await fetch(`${API_URL}/start-mlflow`, { method: 'POST' });
            refreshStatus();
        } catch (e) {
            alert("Failed to connect to backend");
        }
    };

    const stopServer = async () => {
        try {
            await fetch(`${API_URL}/stop/mlflow`, { method: 'POST' });
            refreshStatus();
        } catch (e) {
            alert("Failed to stop server");
        }
    };

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
    };

    const itemVariants = {
        hidden: { y: 20, opacity: 0 },
        visible: { y: 0, opacity: 1 }
    };

    return (
        <motion.div
            className="container"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
        >
            <motion.div variants={itemVariants} style={{ marginBottom: '3rem' }}>
                <h1 style={{ fontSize: '3rem', marginBottom: '1rem' }}>MLflow <span style={{ color: 'var(--primary)' }}>Experiment Tracking</span></h1>
                <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', maxWidth: '800px' }}>
                    Deep insights into model performance, parameters, and versioning across all federated rounds.
                </p>
            </motion.div>

            <motion.div variants={itemVariants} className="premium-card" style={{ marginBottom: '3rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                    <div>
                        <h2 style={{ fontSize: '1.5rem' }}>MLflow Server Control</h2>
                        <p style={{ color: 'var(--text-muted)', marginTop: '0.25rem' }}>Manage the experiment tracking lifespan</p>
                    </div>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '6px 12px',
                        borderRadius: '20px',
                        background: 'var(--glass)',
                        fontSize: '0.75rem',
                        fontWeight: '600',
                        color: status === 'running' ? '#10b981' : 'var(--text-muted)'
                    }}>
                        <span className={`status-dot ${status === 'running' ? 'status-running' : 'status-idle'}`}></span>
                        {status.toUpperCase()}
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '1rem' }}>
                    {status === 'idle' ? (
                        <button onClick={startServer} className="btn-primary">
                            <Play size={18} fill="currentColor" /> Start Server
                        </button>
                    ) : (
                        <button onClick={stopServer} className="btn-primary" style={{ background: 'linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)' }}>
                            <Square size={18} fill="currentColor" /> Stop Server
                        </button>
                    )}
                    {status === 'running' && (
                        <a
                            href={MLFLOW_URL}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn-primary"
                            style={{ background: 'var(--glass)', border: '1px solid var(--glass-border)', boxShadow: 'none' }}
                        >
                            <ExternalLink size={18} /> Open Dashboard
                        </a>
                    )}
                </div>
            </motion.div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
                <motion.div variants={itemVariants} className="premium-card">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', color: 'var(--primary)' }}>
                        <Layers size={20} />
                        <h3 style={{ fontSize: '1.25rem' }}>Active Experiments</h3>
                    </div>
                    <ul style={{ listStyle: 'none', color: 'var(--text-muted)' }}>
                        {[
                            'Client Local Training on CNC machine data',
                            'Federated Learning using the clients',
                            'Global Aggregation Baseline'
                        ].map((exp, i) => (
                            <li key={exp} style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem',
                                padding: '0.75rem 0',
                                borderBottom: i < 2 ? '1px solid var(--glass-border)' : 'none'
                            }}>
                                <Tag size={14} style={{ color: 'var(--primary)' }} />
                                {exp}
                            </li>
                        ))}
                    </ul>
                </motion.div>

                <motion.div variants={itemVariants} className="premium-card">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', color: 'var(--accent)' }}>
                        <BarChart2 size={20} />
                        <h3 style={{ fontSize: '1.25rem' }}>Tracked Metrics</h3>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
                        {['Accuracy', 'Loss', 'Precision', 'Recall', 'F1-Score', 'Latency'].map((metric) => (
                            <span key={metric} style={{
                                background: 'var(--glass)',
                                padding: '0.4rem 0.8rem',
                                borderRadius: '8px',
                                fontSize: '0.85rem',
                                border: '1px solid var(--glass-border)',
                                color: 'var(--text-main)'
                            }}>
                                {metric}
                            </span>
                        ))}
                    </div>
                </motion.div>
            </div>
        </motion.div>
    );
};

export default MLflow;
