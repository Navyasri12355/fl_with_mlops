import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Layout, Server, Database, Activity, GitBranch, Cpu } from 'lucide-react';

const ArchitectureDiagram = () => {
    const [hoveredNode, setHoveredNode] = useState(null);

    const nodes = [
        {
            id: 'ui',
            icon: <Layout size={24} />,
            label: 'React Frontend',
            desc: 'Interactive Dashboard & Controls',
            pos: { x: 10, y: 50 },
            color: '#6366f1'
        },
        {
            id: 'api',
            icon: <Server size={24} />,
            label: 'FastAPI Backend',
            desc: 'Service Management & API Gateway',
            pos: { x: 32, y: 50 },
            color: '#8b5cf6'
        },
        {
            id: 'logic',
            icon: <Cpu size={24} />,
            label: 'Pipeline Logic',
            desc: 'Python process orchestration',
            pos: { x: 58, y: 50 },
            color: '#ec4899'
        },
        {
            id: 'prefect',
            icon: <Database size={24} />,
            label: 'Prefect',
            desc: 'Workflow & Task Scheduling',
            pos: { x: 88, y: 20 },
            color: '#0ea5e9'
        },
        {
            id: 'flower',
            icon: <Activity size={24} />,
            label: 'Flower Server',
            desc: 'Federated Model Aggregation',
            pos: { x: 88, y: 50 },
            color: '#10b981'
        },
        {
            id: 'mlflow',
            icon: <GitBranch size={24} />,
            label: 'MLflow',
            desc: 'Experiment & Metric Tracking',
            pos: { x: 88, y: 80 },
            color: '#f59e0b'
        },
    ];

    const connections = [
        { from: 'ui', to: 'api' },
        { from: 'api', to: 'logic' },
        { from: 'logic', to: 'prefect' },
        { from: 'logic', to: 'flower' },
        { from: 'logic', to: 'mlflow' },
    ];

    return (
        <div style={{ position: 'relative', height: '450px', width: '100%', padding: '20px', background: 'rgba(0,0,0,0.2)', borderRadius: '16px' }}>
            <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                <defs>
                    <marker
                        id="arrowhead"
                        markerWidth="10"
                        markerHeight="7"
                        refX="9"
                        refY="3.5"
                        orient="auto"
                    >
                        <polygon points="0 0, 10 3.5, 0 7" fill="rgba(255, 255, 255, 0.15)" />
                    </marker>
                </defs>
                {connections.map((conn, i) => {
                    const from = nodes.find(n => n.id === conn.from).pos;
                    const to = nodes.find(n => n.id === conn.to).pos;
                    return (
                        <React.Fragment key={i}>
                            <motion.line
                                x1={`${from.x}%`}
                                y1={`${from.y}%`}
                                x2={`${to.x}%`}
                                y2={`${to.y}%`}
                                stroke="var(--glass-border)"
                                strokeWidth="1.5"
                                markerEnd="url(#arrowhead)"
                                initial={{ pathLength: 0, opacity: 0 }}
                                animate={{ pathLength: 1, opacity: 0.3 }}
                                transition={{ duration: 1.5, delay: i * 0.2 }}
                            />
                            {[0, 1, 2].map((particleIndex) => (
                                <motion.circle
                                    key={`${i}-${particleIndex}`}
                                    r="2.5"
                                    fill="var(--primary)"
                                    initial={{ cx: `${from.x}%`, cy: `${from.y}%`, opacity: 0 }}
                                    animate={{
                                        cx: [`${from.x}%`, `${to.x}%`],
                                        cy: [`${from.y}%`, `${to.y}%`],
                                        opacity: [0, 1, 0]
                                    }}
                                    transition={{
                                        duration: 3,
                                        repeat: Infinity,
                                        ease: "linear",
                                        delay: i * 0.4 + particleIndex * 1 // Stagger particles on the same line
                                    }}
                                />
                            ))}
                        </React.Fragment>
                    );
                })}
            </svg>

            {nodes.map((node) => (
                <motion.div
                    key={node.id}
                    onMouseEnter={() => setHoveredNode(node)}
                    onMouseLeave={() => setHoveredNode(null)}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    whileHover={{ scale: 1.05 }}
                    transition={{ type: 'spring', stiffness: 200, damping: 20, delay: 0.5 }}
                    style={{
                        position: 'absolute',
                        left: `${node.pos.x}%`,
                        top: `${node.pos.y}%`,
                        transform: 'translate(-50%, -50%)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        zIndex: 10,
                        cursor: 'pointer'
                    }}
                >
                    <div style={{
                        background: 'var(--bg-card)',
                        border: hoveredNode?.id === node.id ? `2px solid ${node.color}` : '1px solid var(--glass-border)',
                        padding: '1.2rem',
                        borderRadius: '18px',
                        color: node.color,
                        boxShadow: hoveredNode?.id === node.id ? `0 0 25px ${node.color}44` : '0 8px 16px rgba(0,0,0,0.3)',
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        transition: 'all 0.3s ease',
                        backdropFilter: 'blur(12px)',
                        width: '64px',
                        height: '64px'
                    }}>
                        {node.icon}
                    </div>
                    <div style={{ marginTop: '0.75rem', textAlign: 'center', width: '120px' }}>
                        <span style={{
                            display: 'block',
                            fontSize: '0.85rem',
                            fontWeight: '700',
                            color: 'var(--text-main)',
                            textShadow: '0 2px 4px rgba(0,0,0,0.5)',
                            lineHeight: '1.2'
                        }}>
                            {node.label}
                        </span>
                        <AnimatePresence>
                            {hoveredNode?.id === node.id && (
                                <motion.span
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    style={{
                                        display: 'block',
                                        fontSize: '0.65rem',
                                        color: 'var(--text-muted)',
                                        marginTop: '4px',
                                        overflow: 'hidden'
                                    }}
                                >
                                    {node.desc}
                                </motion.span>
                            )}
                        </AnimatePresence>
                    </div>
                </motion.div>
            ))}
        </div>
    );
};

export default ArchitectureDiagram;
