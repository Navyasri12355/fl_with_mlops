import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home as HomeIcon, Cpu, LineChart, Box } from 'lucide-react';
import { motion } from 'framer-motion';

const Navigation = () => {
    const location = useLocation();

    const navItems = [
        { name: 'Home', path: '/', icon: <HomeIcon size={18} /> },
        { name: 'Prefect', path: '/prefect', icon: <Cpu size={18} /> },
        { name: 'MLflow', path: '/mlflow', icon: <LineChart size={18} /> },
    ];

    return (
        <nav className="glass-nav">
            <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 2rem' }}>
                <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', textDecoration: 'none' }}>
                    <div style={{
                        background: 'linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%)',
                        padding: '8px',
                        borderRadius: '10px',
                        display: 'flex',
                        color: 'white'
                    }}>
                        <Box size={24} />
                    </div>
                    <span style={{
                        fontSize: '1.25rem',
                        fontWeight: '800',
                        letterSpacing: '-0.025em',
                        background: 'linear-gradient(135deg, #fff 0%, #94a3b8 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        fontFamily: 'Outfit'
                    }}>
                        FL Dashboard
                    </span>
                </Link>
                <div style={{ display: 'flex', gap: '1rem' }}>
                    {navItems.map((item) => (
                        <Link
                            key={item.path}
                            to={item.path}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem',
                                padding: '0.6rem 1rem',
                                borderRadius: '10px',
                                fontSize: '0.9rem',
                                fontWeight: '500',
                                transition: 'all 0.3s ease',
                                background: location.pathname === item.path ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
                                color: location.pathname === item.path ? 'var(--primary)' : 'var(--text-muted)',
                                textDecoration: 'none'
                            }}
                        >
                            {item.icon}
                            {item.name}
                            {location.pathname === item.path && (
                                <motion.div
                                    layoutId="nav-pill"
                                    style={{
                                        position: 'absolute',
                                        bottom: 0,
                                        left: 0,
                                        right: 0,
                                        height: '2px',
                                        background: 'var(--primary)',
                                        borderRadius: '2px'
                                    }}
                                />
                            )}
                        </Link>
                    ))}
                </div>
            </div>
        </nav>
    );
};

export default Navigation;
