import React, { useState } from 'react';
import { Activity, Network, Settings, Cpu, BrainCircuit } from 'lucide-react';
import TorusVisualizer from './components/TorusVisualizer';
import MetricsCard from './components/MetricsCard';

function App() {
  const [gridSize, setGridSize] = useState(4);
  const [router, setRouter] = useState('grl');
  const [isRunning, setIsRunning] = useState(false);

  // Mock metrics
  const metrics = {
    latency: router === 'xy' ? 12.8 : router === 'odd_even' ? 15.3 : router === 'valiant' ? 25.4 : 10.5,
    throughput: router === 'grl' ? 0.012 : 0.010,
    drops: router === 'grl' ? 0 : 5
  };

  return (
    <div className="container">
      <header style={{ marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 className="text-gradient" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Network size={36} color="var(--accent-primary)" />
            GRL-Torus Visualizer
          </h1>
          <p style={{ color: 'var(--text-secondary)', marginTop: '8px' }}>
            Graph Reinforcement Learning for Adaptive Routing in 2D Torus Optical Interconnects
          </p>
        </div>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn btn-secondary">
            <Settings size={18} />
            Config
          </button>
          <button 
            className={`btn ${isRunning ? 'btn-secondary' : 'btn-primary'}`}
            onClick={() => setIsRunning(!isRunning)}
            style={{ minWidth: '120px' }}
          >
            {isRunning ? 'Pause' : 'Start Simulation'}
          </button>
        </div>
      </header>

      <div className="grid-layout">
        <aside style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Settings size={20} color="var(--accent-secondary)" />
              Simulation Config
            </h3>
            
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)' }}>
                Grid Size ({gridSize}x{gridSize})
              </label>
              <input 
                type="range" 
                min="2" max="8" step="2"
                value={gridSize} 
                onChange={(e) => setGridSize(parseInt(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
              />
            </div>

            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)' }}>
                Routing Algorithm
              </label>
              <select 
                value={router} 
                onChange={(e) => setRouter(e.target.value)}
                style={{ 
                  width: '100%', 
                  padding: '10px', 
                  borderRadius: '8px', 
                  background: 'rgba(0,0,0,0.2)',
                  border: '1px solid var(--border-color)',
                  color: 'white'
                }}
              >
                <option value="xy">XY Routing (Baseline)</option>
                <option value="odd_even">Odd-Even Routing</option>
                <option value="valiant">Valiant Load Balancing</option>
                <option value="gnn">Supervised GNN</option>
                <option value="grl">GRL (Ours)</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <MetricsCard 
              title="Avg Latency" 
              value={`${metrics.latency.toFixed(1)} ns`} 
              icon={<Activity size={24} color="var(--accent-primary)" />} 
              trend={router === 'grl' ? "-18%" : "+0%"}
              trendUp={false}
            />
            <MetricsCard 
              title="Throughput" 
              value={`${metrics.throughput.toFixed(3)} pps`} 
              icon={<Cpu size={24} color="var(--accent-secondary)" />} 
              trend={router === 'grl' ? "+12%" : "+0%"}
              trendUp={true}
            />
            <MetricsCard 
              title="Dropped Packets" 
              value={metrics.drops.toString()} 
              icon={<BrainCircuit size={24} color="var(--accent-warning)" />} 
            />
          </div>
        </aside>

        <main className="glass-panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden' }}>
          <TorusVisualizer gridSize={gridSize} isRunning={isRunning} routerType={router} />
          
          <div style={{ position: 'absolute', bottom: '20px', left: '20px', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Optical Node 
            <span style={{ display: 'inline-block', width: '12px', height: '12px', borderRadius: '50%', background: 'var(--border-highlight)', margin: '0 8px', border: '1px solid var(--accent-primary)' }}></span>
            Active Link
            <span style={{ display: 'inline-block', width: '20px', height: '3px', background: 'var(--accent-primary)', margin: '0 8px', boxShadow: '0 0 8px var(--accent-primary-glow)' }}></span>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
