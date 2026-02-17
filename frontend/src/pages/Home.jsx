import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Square, Info, ShieldCheck, Activity, ChevronRight, Share2, Cpu } from 'lucide-react';
import ArchitectureDiagram from '../components/ArchitectureDiagram';
import { useStatus } from '../context/StatusContext';
import { supabase } from '../supabase';
import { useAuth } from '../context/AuthContext';

const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
        y: 0,
        opacity: 1,
        transition: { duration: 0.5, ease: 'easeOut' }
    }
};

const Home = () => {
    const { status: globalStatus, refreshStatus } = useStatus();
    const { user } = useAuth();
    const status = globalStatus.pipeline;
    const [showArch, setShowArch] = useState(false);
    const [bestAccuracy, setBestAccuracy] = useState(null);

    const fetchBestAccuracy = async () => {
        if (!user) return;
        const { data, error } = await supabase
            .from('runs')
            .select('accuracy')
            .eq('user_id', user.id)
            .order('accuracy', { ascending: false })
            .limit(1);

        if (data && data.length > 0) {
            setBestAccuracy(data[0].accuracy);
        }
    };

    useEffect(() => {
        fetchBestAccuracy();
    }, [user]);

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


    const [lastStatus, setLastStatus] = useState(status);

    useEffect(() => {
        // Detect pipeline completion
        if (lastStatus === 'running' && status === 'idle') {
            handlePipelineCompletion();
        }
        setLastStatus(status);
    }, [status]);

    const handlePipelineCompletion = async () => {
        console.log("Pipeline completion detected, fetching results...");
        try {
            const res = await fetch('http://localhost:8000/latest-result');
            const data = await res.json();
            console.log("Latest result data:", data);

            if (data && !data.error && user) {
                console.log("Recording run results to Supabase...");
                const { error: insertError } = await supabase.from('runs').insert({
                    user_id: user.id,
                    accuracy: data.accuracy,
                    round_count: data.rounds,
                    duration: '4 rounds' // Placeholder
                });
                if (insertError) console.error("Supabase insert error:", insertError);

                fetchBestAccuracy();
            } else {
                console.warn("Pipeline completed but no valid data found or user not logged in", data);
            }
        } catch (e) {
            console.error("Failed to record completion:", e);
        }
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
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                        <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: '10px', borderRadius: '12px', color: 'var(--success)', width: 'fit-content' }}>
                            <ShieldCheck size={24} />
                        </div>
                        <button
                            onClick={() => {
                                handlePipelineCompletion();
                                fetchBestAccuracy();
                            }}
                            className="btn-primary"
                            style={{ padding: '6px 12px', fontSize: '0.75rem', background: 'var(--glass)', border: '1px solid var(--glass-border)', boxShadow: 'none' }}
                        >
                            Refresh
                        </button>
                    </div>
                    <h2 style={{ fontSize: '1.5rem' }}>Best Model</h2>
                    <p style={{ marginTop: '0.5rem', minHeight: '3rem' }}>
                        Automatically tracked and persisted in <code>best_model/</code>.
                    </p>
                    <div style={{ marginTop: '2rem', padding: '1rem', background: 'var(--glass)', borderRadius: '12px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                            <span>Global Accuracy</span>
                            <span style={{ color: 'var(--success)', fontWeight: '700' }}>
                                {bestAccuracy ? `${bestAccuracy.toFixed(1)}%` : 'N/A'}
                            </span>
                        </div>
                        <div style={{ height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', marginTop: '0.5rem', overflow: 'hidden' }}>
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: bestAccuracy ? `${bestAccuracy}%` : '0%' }}
                                transition={{ duration: 1.5, ease: 'easeOut' }}
                                style={{ height: '100%', background: 'var(--success)' }}
                            ></motion.div>
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
