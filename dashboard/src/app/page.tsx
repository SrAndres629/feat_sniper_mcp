"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Target, Activity, Zap, Shield, Database, Cpu, Search } from 'lucide-react';
import { supabase } from '@/lib/supabase';

// --- COMPONENTS ---

const GlassCard = ({ children, className = "" }: { children: React.ReactNode, className?: string }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className={`glass p-6 rounded-xl relative overflow-hidden ${className}`}
  >
    <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-aqua/30 to-transparent" />
    {children}
  </motion.div>
);

const MetricBox = ({ label, value, icon: Icon, color = "aqua" }: any) => (
  <div className="flex items-center space-x-4 p-3 border border-white/5 rounded-lg bg-white/5">
    <div className={`p-2 rounded-md bg-${color}/10`}>
      <Icon className={`w-5 h-5 text-${color}`} />
    </div>
    <div>
      <p className="text-[10px] text-gray-400 uppercase tracking-widest">{label}</p>
      <p className="text-lg font-mono font-bold">{value}</p>
    </div>
  </div>
);

// --- MAIN PAGE ---

export default function Dashboard() {
  const [signals, setSignals] = useState<any[]>([]);
  const [score, setScore] = useState(92);
  const [status, setStatus] = useState("OPERATIONAL");

  useEffect(() => {
    // Real-time signal subscription
    const channel = supabase
      .channel('live-signals')
      .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'sniper_signals' }, payload => {
        setSignals(prev => [payload.new, ...prev].slice(0, 10));
        if (payload.new.confidence) setScore(payload.new.confidence);
      })
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, []);

  return (
    <main className="min-h-screen bg-background text-foreground p-4 md:p-8 font-sans selection:bg-aqua/30">
      {/* Background Decorative Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-aqua/5 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-gold/5 blur-[120px] rounded-full" />
      </div>

      <div className="max-w-7xl mx-auto space-y-6 relative z-10">

        {/* HEADER */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center glass p-6 rounded-2xl border-aqua/20">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-aqua/20 rounded-lg flex items-center justify-center border border-aqua/30">
              <Target className="w-8 h-8 text-aqua text-glow-aqua" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tighter flex items-center">
                FEAT <span className="text-aqua mx-2">SNIPER</span>
                <span className="text-[10px] px-2 py-0.5 border border-gold/50 text-gold rounded ml-2 uppercase tracking-tighter">NEXUS v1.0</span>
              </h1>
              <p className="text-xs text-gray-500 font-mono tracking-tight uppercase">Institutional AI Trading Terminal</p>
            </div>
          </div>

          <div className="mt-4 md:mt-0 flex items-center space-x-6">
            <div className="flex flex-col items-end">
              <span className="text-[10px] text-gray-400 uppercase tracking-widest">System Status</span>
              <span className="text-green-sniper flex items-center text-sm font-bold font-mono">
                <span className="w-2 h-2 bg-green-sniper rounded-full mr-2 animate-pulse" />
                {status}
              </span>
            </div>
            <button className="p-2 rounded-full border border-white/10 hover:bg-white/5 transition-colors">
              <Search className="w-5 h-5 text-gray-400" />
            </button>
          </div>
        </header>

        {/* GRID LAYOUT */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* LEFT: MONITORING */}
          <div className="lg:col-span-8 space-y-6">
            <GlassCard className="h-[400px] flex flex-col justify-center items-center">
              {/* Central Gauge Mockup */}
              <div className="relative w-64 h-64 flex items-center justify-center">
                <svg className="w-full h-full transform -rotate-90">
                  <circle cx="50%" cy="50%" r="45%" className="stroke-white/5 fill-transparent" strokeWidth="8" />
                  <circle
                    cx="50%" cy="50%" r="45%"
                    className="stroke-aqua fill-transparent transition-all duraton-1000"
                    strokeWidth="8"
                    strokeDasharray="283"
                    strokeDashoffset={283 - (283 * score) / 100}
                    strokeLinecap="round"
                    style={{ filter: "drop-shadow(0 0 8px rgba(0, 242, 255, 0.5))" }}
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-5xl font-bold tracking-tighter text-glow-aqua">{score}%</span>
                  <span className="text-[10px] text-gray-400 uppercase tracking-widest mt-2">FEAT Confidence</span>
                </div>
              </div>

              <div className="mt-8 flex space-x-8">
                <div className="text-center">
                  <p className="text-[10px] text-gray-500 uppercase tracking-widest">Verdict</p>
                  <p className="text-lg font-bold text-aqua uppercase">Strong Accumulation</p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] text-gray-500 uppercase tracking-widest">Risk/Reward</p>
                  <p className="text-lg font-bold text-gold">1 : 2.5</p>
                </div>
              </div>
            </GlassCard>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricBox label="ZMQ Bridge" value="1.2ms" icon={Zap} />
              <MetricBox label="RAG Memory" icon={Database} value="Indexed" color="gold" />
              <MetricBox label="Offloading" value="Active" icon={Cpu} />
              <MetricBox label="Protection" value="Breaker ON" icon={Shield} color="green-sniper" />
            </div>
          </div>

          {/* RIGHT: SIGNALS */}
          <div className="lg:col-span-4 space-y-6">
            <GlassCard className="h-full min-h-[500px]">
              <div className="flex items-center justify-between mb-6">
                <h3 className="font-bold flex items-center">
                  <Activity className="w-4 h-4 mr-2 text-aqua" />
                  LIVE SIGNALS
                </h3>
                <span className="text-[10px] px-2 py-0.5 bg-red-sniper/20 text-red-sniper rounded">REALTIME</span>
              </div>

              <div className="space-y-4 max-h-[450px] overflow-y-auto pr-2 custom-scrollbar">
                <AnimatePresence>
                  {signals.length === 0 ? (
                    <p className="text-gray-500 text-sm font-mono text-center mt-12 italic opacity-50">Awaiting market events...</p>
                  ) : (
                    signals.map((sig, i) => (
                      <motion.div
                        key={sig.id || i}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="p-4 rounded-lg bg-white/5 border border-white/5 hover:border-aqua/30 transition-all cursor-pointer group"
                      >
                        <div className="flex justify-between items-start">
                          <div>
                            <p className="font-bold text-lg group-hover:text-aqua transition-colors">{sig.symbol}</p>
                            <p className="text-[10px] text-gray-500 font-mono italic">{new Date(sig.created_at).toLocaleTimeString()}</p>
                          </div>
                          <span className={`px-2 py-1 rounded text-[10px] font-bold ${sig.action === 'BUY' ? 'bg-green-sniper/20 text-green-sniper' : 'bg-red-sniper/20 text-red-sniper'
                            }`}>
                            {sig.action}
                          </span>
                        </div>
                        <div className="mt-3 flex justify-between items-center text-xs font-mono">
                          <span className="text-gray-400">P: {sig.price}</span>
                          <span className="text-aqua">{sig.confidence}% CONF</span>
                        </div>
                      </motion.div>
                    ))
                  )}
                </AnimatePresence>
              </div>
            </GlassCard>
          </div>

        </div>
      </div>
    </main>
  );
}
