import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Square, Info, ShieldCheck, Activity, ChevronRight, Share2, Cpu } from 'lucide-react';
import ArchitectureDiagram from '../components/ArchitectureDiagram';
import { useStatus } from '../context/StatusContext';

const Home = () => {
    const { status: globalStatus, refreshStatus } = useStatus();
    const status = globalStatus.pipeline;
    const [showArch, setShowArch] = useState(false);

    const runPipeline = async () => {
        try {
            await fetch('http://localhost:8000/run-pipeline', { method: 'POST' });
            refreshStatus();
        } catch (e) {
            alert("Failed to connect to backend");
        }
    };

    const stopPipeline = async () => {
        try {
            await fetch('http://localhost:8000/stop/pipeline', { method: 'POST' });
            refreshStatus();
        } catch (e) {
            alert("Failed to stop pipeline");
        }
    };

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: { staggerChildren: 0.1 }
        }
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
                <h1 style={{ fontSize: '3.5rem', marginBottom: '1rem', letterSpacing: '-0.02em' }}>
                    Federated Learning <span style={{ color: 'var(--primary)' }}>Dashboard</span>
                </h1>
                <p style={{ fontSize: '1.25rem', maxWidth: '800px', color: 'var(--text-muted)' }}>
                    Manage your distributed AI workloads with a high-performance orchestration engine.
                    Real-time tracking, automated aggregation, and seamless MLOps integration.
                </p>
            </motion.div>

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
                gap: '2rem',
                marginBottom: '3rem'
            }}>
                <motion.div variants={itemVariants} className="premium-card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                        <div style={{ background: 'rgba(99, 102, 241, 0.1)', padding: '10px', borderRadius: '12px', color: 'var(--primary)' }}>
                            <Activity size={24} />
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
                    <h2 style={{ fontSize: '1.5rem' }}>Pipeline Control</h2>
                    <p style={{ marginTop: '0.5rem', minHeight: '3rem' }}>
                        Live orchestration of communication rounds and global model aggregation.
                    </p>
                    <div style={{ marginTop: '2rem' }}>
                        {status === 'idle' ? (
                            <button onClick={runPipeline} className="btn-primary" style={{ width: '100%', justifyContent: 'center' }}>
                                <Play size={18} fill="currentColor" /> Start Pipeline
                            </button>
                        ) : (
                            <button onClick={stopPipeline} className="btn-primary" style={{ width: '100%', justifyContent: 'center', background: 'linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)' }}>
                                <Square size={18} fill="currentColor" /> Terminate Run
                            </button>
                        )}
                    </div>
                </motion.div>

                <motion.div variants={itemVariants} className="premium-card">
                    <div style={{ background: 'rgba(168, 85, 247, 0.1)', padding: '10px', borderRadius: '12px', color: 'var(--accent)', marginBottom: '1.5rem', width: 'fit-content' }}>
                        <Share2 size={24} />
                    </div>
                    <h2 style={{ fontSize: '1.5rem' }}>Infrastructure</h2>
                    <p style={{ marginTop: '0.5rem', minHeight: '3rem' }}>
                        Prefect workflows and MLflow experiment tracking integration.
                    </p>
                    <button
                        onClick={() => setShowArch(!showArch)}
                        className="btn-primary"
                        style={{ marginTop: '2rem', width: '100%', justifyContent: 'center', background: 'var(--glass)', border: '1px solid var(--glass-border)', boxShadow: 'none' }}
                    >
                        <Info size={18} /> {showArch ? 'Hide Architecture' : 'System Insights'}
                    </button>
                </motion.div>

                <motion.div variants={itemVariants} className="premium-card">
                    <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: '10px', borderRadius: '12px', color: 'var(--success)', marginBottom: '1.5rem', width: 'fit-content' }}>
                        <ShieldCheck size={24} />
                    </div>
                    <h2 style={{ fontSize: '1.5rem' }}>Best Model</h2>
                    <p style={{ marginTop: '0.5rem', minHeight: '3rem' }}>
                        Automatically tracked and persisted in <code>best_model/</code>.
                    </p>
                    <div style={{ marginTop: '2rem', padding: '1rem', background: 'var(--glass)', borderRadius: '12px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                            <span>Global Accuracy</span>
                            <span style={{ color: 'var(--success)', fontWeight: '700' }}>92.4%</span>
                        </div>
                        <div style={{ height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', marginTop: '0.5rem', overflow: 'hidden' }}>
                            <motion.div initial={{ width: 0 }} animate={{ width: '92.4%' }} transition={{ duration: 1.5, ease: 'easeOut' }} style={{ height: '100%', background: 'var(--success)' }}></motion.div>
                        </div>
                    </div>
                </motion.div>
            </div>

            <AnimatePresence>
                {showArch && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        style={{ overflow: 'hidden' }}
                    >
                        <div className="premium-card">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2rem' }}>
                                <Cpu size={20} className="text-primary" />
                                <h2 style={{ fontSize: '1.25rem' }}>Full Stack Architecture</h2>
                            </div>
                            <ArchitectureDiagram />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

export default Home;
