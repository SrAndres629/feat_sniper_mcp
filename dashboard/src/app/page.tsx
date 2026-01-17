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
    <div className="absolute top-0 left-0 w-full h-px bg-linear-to-r from-transparent via-aqua/30 to-transparent" />
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
  const [neuralState, setNeuralState] = useState<any>({
    win_confidence: 0,
    alpha_multiplier: 1.0,
    volatility_regime: 0,
    price: 0,
    uncertainty: 0,
    pvp: { energy: 0, dist_poc: 0, skew: 0, entropy: 0 }
  });
  const [expandedSignal, setExpandedSignal] = useState<string | null>(null);
  const [score, setScore] = useState(0);
  const [status, setStatus] = useState("SYNCING");
  const [pnl, setPnl] = useState({ daily: 0, total: 0, hourly: 0, velocity: 0 });
  const [history, setHistory] = useState<any[]>([]);
  const [twinEngine, setTwinEngine] = useState({ scalp: 0, swing: 0, mode: "NEURAL" });

  useEffect(() => {
    // 4-Head Neural Telemetry subscription
    const neuralChannel = supabase
      .channel('neural-updates')
      .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'neural_signals' }, payload => {
        const data = payload.new;
        setNeuralState({
          win_confidence: data.alpha_confidence || 0,
          alpha_multiplier: data.alpha_multiplier || 1.0,
          volatility_regime: data.volatility_regime || 0,
          price: data.price || 0,
          uncertainty: data.metadata?.uncertainty || 0,
          pvp: {
            energy: data.energy_score || 0,
            dist_poc: data.dist_poc || 0,
            skew: data.skew || 0,
            entropy: data.entropy || 0
          }
        });
        setScore(Math.round((data.alpha_confidence || 0) * 100));
        setStatus("ACTIVE");
      })
      .subscribe();

    // Fetch Financial Performance every 30 seconds
    const fetchPerformance = async () => {
      try {
        const { data: vault } = await supabase.from('vault_state').select('*').single();
        if (vault) {
          setPnl(prev => ({ ...prev, total: vault.trading_capital }));
        }
      } catch (e) {
        console.error("Failed to fetch financial performance", e);
      }
    };

    const interval = setInterval(fetchPerformance, 30000);
    fetchPerformance();

    return () => {
      supabase.removeChannel(channel);
      supabase.removeChannel(neuralChannel);
      clearInterval(interval);
    };
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
            <GlassCard className="h-[400px] flex flex-col justify-center items-center relative">
              <div className="absolute top-4 right-4 text-right">
                <p className="text-[10px] text-gray-500 uppercase tracking-widest">Financial Nexus</p>
                <p className={`text-xl font-mono font-bold ${pnl.daily >= 0 ? 'text-green-sniper' : 'text-red-sniper'}`}>
                  ${pnl.daily.toLocaleString()} <span className="text-[10px]">USD/DAY</span>
                </p>
              </div>

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
                  <span className="text-[10px] text-gray-400 uppercase tracking-widest mt-2">{neuralState.win_confidence > 0.8 ? 'DOCTORAL ALPHA' : 'NEURAL FLOW'}</span>
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
              <MetricBox label="Neural Alpha" value={`x${neuralState.alpha_multiplier.toFixed(2)}`} icon={Zap} color="gold" />
              <MetricBox label="Vol Regime" value={neuralState.volatility_regime.toFixed(2)} icon={Activity} color="white" />
              <MetricBox label="Entropy" value={neuralState.pvp.entropy.toFixed(2)} icon={Database} color="aqua" />
              <MetricBox label="Uncertainty" value={neuralState.uncertainty.toFixed(3)} icon={Shield} color="red-sniper" />
            </div>

            {/* TRADE JOURNAL */}
            <GlassCard className="min-h-[300px]">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold flex items-center text-sm">
                  <Database className="w-4 h-4 mr-2 text-gold" />
                  TACTICAL TRADE JOURNAL
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="text-gray-500 border-b border-white/5">
                      <th className="text-left pb-2">TIME</th>
                      <th className="text-left pb-2">ASSET</th>
                      <th className="text-left pb-2">TYPE</th>
                      <th className="text-left pb-2">SETUP (FEAT)</th>
                      <th className="text-right pb-2">PROFIT</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {/* Placeholder Journal Data */}
                    <tr>
                      <td className="py-2 text-gray-400">04:12:00</td>
                      <td className="py-2 font-bold text-aqua">XAUUSD</td>
                      <td className="py-2"><span className="text-green-sniper">BUY</span></td>
                      <td className="py-2 text-gray-400">OB+FVG (H1)</td>
                      <td className="py-2 text-right text-green-sniper">+$420.00</td>
                    </tr>
                    <tr>
                      <td className="py-2 text-gray-400">03:55:20</td>
                      <td className="py-2 font-bold text-aqua">BTCUSD</td>
                      <td className="py-2"><span className="text-red-sniper">SELL</span></td>
                      <td className="py-2 text-right text-red-sniper">-$120.50</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </GlassCard>
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
                        onClick={() => setExpandedSignal(expandedSignal === sig.id ? null : sig.id)}
                      >
                        <div className="flex justify-between items-start">
                          <div>
                            <p className="font-bold text-lg group-hover:text-aqua transition-colors">{sig.symbol}</p>
                            <p className="text-[10px] text-gray-400 font-mono italic">{new Date(sig.created_at).toLocaleTimeString()}</p>
                          </div>
                          <span className={`px-2 py-1 rounded text-[10px] font-bold ${(sig.direction || sig.action) === 'BUY' ? 'bg-green-sniper/20 text-green-sniper' : 'bg-red-sniper/20 text-red-sniper'
                            }`}>
                            {sig.direction || sig.action}
                          </span>
                        </div>
                        <div className="mt-3 flex justify-between items-center text-xs font-mono">
                          <span className="text-gray-400">P: {sig.entry_price || sig.price}</span>
                          <span className="text-aqua">{(sig.confidence <= 1 ? sig.confidence * 100 : sig.confidence).toFixed(0)}% CONF</span>
                        </div>

                        {/* REASONING SECTION */}
                        <AnimatePresence>
                          {expandedSignal === sig.id && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              className="mt-4 pt-4 border-t border-white/10 overflow-hidden"
                            >
                              <div className="flex items-center space-x-2 mb-2 text-gold">
                                <Search className="w-3 h-3" />
                                <span className="text-[10px] uppercase tracking-widest font-bold">FEAT Reasoning</span>
                              </div>
                              <p className="text-xs text-gray-300 font-mono leading-relaxed bg-black/20 p-2 rounded">
                                {sig.explanation || sig.engineer_diagnosis || sig.metadata?.explanation || "Analyzing market microstructure for definitive bias..."}
                              </p>
                              {sig.metadata?.top_drivers && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {sig.metadata.top_drivers.map((driver: string, idx: number) => (
                                    <span key={idx} className="text-[8px] px-1.5 py-0.5 bg-aqua/10 text-aqua rounded border border-aqua/20">
                                      {driver}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </motion.div>
                          )}
                        </AnimatePresence>
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
