import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { Leaf, Activity, Cpu, BookOpen, Layers, ShieldCheck, Menu, X } from 'lucide-react';
import { healthCheck } from '../services/api';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    healthCheck()
      .then((data) => {
        if (data.status === 'operational') {
          setBackendStatus('online');
        } else {
          setBackendStatus('offline');
        }
      })
      .catch(() => setBackendStatus('offline'));
  }, []);

  const navItems = [
    { name: 'Home', path: '/' },
    { name: 'Disease Detection', path: '/detect' },
    { name: 'Disease Info', path: '/diseases' },
    { name: 'Supported Crops', path: '/crops' },
    { name: 'Model Performance', path: '/performance' },
    { name: 'About Model', path: '/about' },
  ];

  return (
    <nav className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <NavLink to="/" className="flex items-center gap-2.5 group">
              <div className="w-10 h-10 rounded-xl bg-emerald-600 flex items-center justify-center text-white shadow-md group-hover:bg-emerald-700 transition">
                <Leaf className="w-6 h-6" />
              </div>
              <div>
                <span className="text-lg font-bold text-slate-900 tracking-tight block leading-tight">
                  AgriVision <span className="text-emerald-600 font-extrabold">AI</span>
                </span>
                <span className="text-xs text-slate-500 font-medium hidden sm:block">
                  ResNet-50 Multi-Crop Disease Classifier
                </span>
              </div>
            </NavLink>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1 lg:space-x-2">
            {navItems.map((item) => (
              <NavLink
                key={item.name}
                to={item.path}
                className={({ isActive }) =>
                  `px-3.5 py-2 rounded-lg text-sm font-semibold transition ${
                    isActive
                      ? 'bg-emerald-50 text-emerald-700 font-bold'
                      : 'text-slate-600 hover:text-emerald-600 hover:bg-slate-50'
                  }`
                }
              >
                {item.name}
              </NavLink>
            ))}

            {/* Backend health pill */}
            <div className="ml-3 pl-3 border-l border-slate-200 flex items-center gap-1.5">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  backendStatus === 'online'
                    ? 'bg-emerald-500 animate-pulse'
                    : backendStatus === 'checking'
                    ? 'bg-amber-400'
                    : 'bg-rose-500'
                }`}
              />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                {backendStatus === 'online' ? 'API Online' : backendStatus === 'checking' ? 'Checking' : 'API Offline'}
              </span>
            </div>
          </div>

          {/* Mobile hamburger */}
          <div className="flex items-center md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="p-2 rounded-lg text-slate-600 hover:text-emerald-600 hover:bg-slate-100 focus:outline-none"
            >
              {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6 text-slate-700" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="md:hidden border-t border-slate-200 bg-white px-4 pt-2 pb-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.name}
              to={item.path}
              onClick={() => setIsOpen(false)}
              className={({ isActive }) =>
                `block px-3 py-2 rounded-md text-base font-medium ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-700 font-bold'
                    : 'text-slate-700 hover:bg-slate-50 hover:text-emerald-600'
                }`
              }
            >
              {item.name}
            </NavLink>
          ))}
          <div className="pt-2 flex items-center gap-2 px-3 text-xs text-slate-500">
            <span
              className={`w-2 h-2 rounded-full ${
                backendStatus === 'online' ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            />
            Backend: {backendStatus}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
