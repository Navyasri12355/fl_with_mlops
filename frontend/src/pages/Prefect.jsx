import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Cpu, Play, Square, ExternalLink, Activity, Database, Clock } from 'lucide-react';
import { useStatus } from '../context/StatusContext';
import { supabase } from '../supabase';
import { useAuth } from '../context/AuthContext';

const Prefect = () => {
    const { status: globalStatus, refreshStatus } = useStatus();
    const { user } = useAuth();
    const status = globalStatus.prefect;
    const [metrics, setMetrics] = useState({
        runsToday: 0,
        successRate: 0,
        avgDuration: '0s'
    });

    const fetchMetrics = async () => {
        if (!user) return;

        const startOfDay = new Date();
        startOfDay.setHours(0, 0, 0, 0);

        // Runs today
        const { count: runsToday } = await supabase
            .from('runs')
            .select('*', { count: 'exact', head: true })
            .eq('user_id', user.id)
            .gte('timestamp', startOfDay.toISOString());

        // Overall stats
        const { data: allRuns } = await supabase
            .from('runs')
            .select('accuracy, duration')
            .eq('user_id', user.id);

        if (allRuns && allRuns.length > 0) {
            const successCount = allRuns.filter(r => r.accuracy > 0).length;
            const successRate = (successCount / allRuns.length) * 100;

            // Just a placeholder for avg duration since it's a string in our mock/example
            setMetrics({
                runsToday: runsToday || 0,
                successRate: successRate.toFixed(0),
                avgDuration: '4m 12s'
            });
        }
    };

    useEffect(() => {
        fetchMetrics();
    }, [user]);

    const startServer = async () => {
        try {
            await fetch('http://localhost:8000/start-prefect', { method: 'POST' });
            refreshStatus();
        } catch (e) {
            alert("Failed to connect to backend");
        }
    };

    const stopServer = async () => {
        try {
            await fetch('http://localhost:8000/stop/prefect', { method: 'POST' });
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
                <h1 style={{ fontSize: '3rem', marginBottom: '1rem' }}>Prefect <span style={{ color: 'var(--primary)' }}>Orchestration</span></h1>
                <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', maxWidth: '800px' }}>
                    Monitor and manage your machine learning workflows with high-visibility pipeline orchestration.
                </p>
            </motion.div>

            <motion.div variants={itemVariants} className="premium-card" style={{ marginBottom: '3rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                    <div>
                        <h2 style={{ fontSize: '1.5rem' }}>Prefect Server Control</h2>
                        <p style={{ color: 'var(--text-muted)', marginTop: '0.25rem' }}>Manage the central orchestration engine</p>
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
                            href="http://localhost:4200"
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
                        <Activity size={20} />
                        <h3 style={{ fontSize: '1.25rem' }}>Active Flows</h3>
                    </div>
                    <ul style={{ listStyle: 'none', color: 'var(--text-muted)' }}>
                        {['distributed_fl_pipeline', 'data_consolidation_flow', 'model_evaluation_flow'].map((flow, i) => (
                            <li key={flow} style={{
                                borderBottom: i < 2 ? '1px solid var(--glass-border)' : 'none',
                                padding: '0.75rem 0',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem'
                            }}>
                                <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--primary)' }}></div>
                                {flow}
                            </li>
                        ))}
                    </ul>
                </motion.div>

                <motion.div variants={itemVariants} className="premium-card">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', color: 'var(--accent)' }}>
                        <Clock size={20} />
                        <h3 style={{ fontSize: '1.25rem' }}>Workflow Metrics</h3>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', color: 'var(--text-muted)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>Runs Today</span>
                            <span style={{ color: 'var(--text-main)', fontWeight: '600' }}>{metrics.runsToday}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>Success Rate</span>
                            <span style={{ color: 'var(--success)', fontWeight: '600' }}>{metrics.successRate}%</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>Avg Duration</span>
                            <span style={{ color: 'var(--text-main)', fontWeight: '600' }}>{metrics.avgDuration}</span>
                        </div>
                    </div>
                </motion.div>
            </div>
        </motion.div>
    );
};

export default Prefect;
