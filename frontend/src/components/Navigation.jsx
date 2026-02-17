import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Home as HomeIcon, Cpu, LineChart, Box, LogOut, User } from 'lucide-react';
import { motion } from 'framer-motion';
import { useAuth } from '../context/AuthContext';

const Navigation = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { user, signOut } = useAuth();

    const navItems = [
        { name: 'Home', path: '/', icon: <HomeIcon size={18} />, protected: true },
        { name: 'Prefect', path: '/prefect', icon: <Cpu size={18} />, protected: true },
        { name: 'MLflow', path: '/mlflow', icon: <LineChart size={18} />, protected: true },
    ];

    const handleLogout = async () => {
        await signOut();
        navigate('/login');
    };

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
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    {user ? (
                        <>
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
                                        textDecoration: 'none',
                                        position: 'relative'
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
                            <div style={{ width: '1px', height: '24px', background: 'var(--glass-border)', margin: '0 0.5rem' }}></div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--text-main)', fontSize: '0.9rem' }}>
                                <div style={{
                                    width: '32px',
                                    height: '32px',
                                    borderRadius: '50%',
                                    background: 'var(--glass)',
                                    border: '1px solid var(--glass-border)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    color: 'var(--primary)'
                                }}>
                                    <User size={16} />
                                </div>
                                <span style={{ maxWidth: '120px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    {user.email.split('@')[0]}
                                </span>
                            </div>
                            <button
                                onClick={handleLogout}
                                style={{
                                    background: 'transparent',
                                    border: 'none',
                                    color: '#f87171',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.4rem',
                                    fontSize: '0.9rem',
                                    padding: '0.5rem',
                                    borderRadius: '8px',
                                    transition: 'all 0.2s ease'
                                }}
                                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)'}
                                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                            >
                                <LogOut size={16} />
                            </button>
                        </>
                    ) : (
                        <div style={{ display: 'flex', gap: '1rem' }}>
                            <Link to="/login" style={{
                                padding: '0.6rem 1.2rem',
                                color: 'var(--text-main)',
                                textDecoration: 'none',
                                fontSize: '0.9rem',
                                fontWeight: '500'
                            }}>
                                Login
                            </Link>
                            <Link to="/signup" className="btn-primary" style={{ padding: '0.6rem 1.2rem', fontSize: '0.9rem' }}>
                                Sign Up
                            </Link>
                        </div>
                    )}
                </div>
            </div>
        </nav>
    );
};

export default Navigation;
